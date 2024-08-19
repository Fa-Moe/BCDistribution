import pandas
import numpy
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
import copy
import os
import json


class BCDistribution:
    """
    The BCDistribution class holds methods to analyze numeric data and generate Discrete Generalized Beta Distributions.
    All attributes are instance attributes. When creating an instance, all attributes but df_abundance are None.
    Attributes:
        df_abundance: Data frame used to initialize the object.
        rank_data: Contains the sorted data frame.
        rank_data_colname: Stores the name of the sorted column. It's used in the plot() and report() methods.
        param_data_frame: A data frame with 'pre_numerator' to fit the model and resulting predicted values.
        param_data_matrix: A 3x3 data frame with the parameters of the model with confidence interval
        param_data_model: The RegressionResults object from the OLS model
        param_data_summary: A summary of the OLS model
        model_data: The predicted abundance (y) value.
        gof_data_model: The name of the calculated goodness of fit.
        gof_data_value: The value (number) of the calculated goodness of fit.
        plot_data: The path to the generated graph.
        report_data: A data frame with the goodness of fit value and paramaters obtained.
    """
    def __init__(self, df_abundance):
        """
        Initialize a BCDistribution class object.
        :param df_abundance: The data frame to hold. Can be a dictionary or a pandas Data Frame.
        """
        self.df_abundance = pandas.DataFrame(df_abundance)
        self.rank_data = None
        self.rank_data_colname = None
        self.param_data_frame = None
        self.param_data_matrix = None
        self.param_data_model = None
        self.param_data_summary = None
        self.model_data = None
        self.gof_data_model = None
        self.gof_data_value = None
        self.plot_data = None
        self.report_data = None

    def rank(self, column_rank, rank_threshold=0, **_):
        """
        The rank() method has to be called on a generated BCDistribution object.
        It will sort the contained dataframe by the column_rank column.
        :param column_rank: Can be either a string containing the name of the column or an int that is its index
        :param rank_threshold: Optional. Integer. Discards rows if their column_rank value is equal to or lower than it.
        :param _: Internal. Used in the caller() method to ignore unused keyword arguments.
        :return: It returns the given BCDistribution object, but with 2 attributes updated:
                 rank_data: Contains the sorted data frame.
                 rank_data_colname: Stores the name of the sorted column. It's used in the plot() and report() methods.
        """
        df_abundance = self.df_abundance
        df_abundance = df_abundance.reset_index(drop=True)
        try:
            df_abundance = df_abundance.sort_values(by=[df_abundance.columns[column_rank]], ascending=False)
        except IndexError:
            column_rank = numpy.asarray(df_abundance.columns.values == column_rank).nonzero()[0][0]
            df_abundance = df_abundance.sort_values(by=[df_abundance.columns[column_rank]], ascending=False)

        df_abundance = df_abundance[df_abundance.iloc[:, column_rank] > rank_threshold]
        self.rank_data_colname = self.df_abundance.columns[column_rank]
        df_abundance = df_abundance.rename(columns={df_abundance.columns[column_rank]: 'abundance'})
        ranking = pandas.Series(range(1, df_abundance.shape[0]+1))
        ranking.name = 'BC_rank'
        ranking.index = df_abundance.index
        df_abundance = df_abundance.join(ranking)
        self.rank_data = df_abundance
        return self

    def param(self, confidence_interval=0.95, **_):
        """
        The param() method has to be called after the rank() method has been called.
        Estimates the parameters of the Discrete Generalized Beta Distribution (also known as the Beta-Cocho
        distribution, BCD) proposed in Martinez-Mekler et al. (2009) DOI:10.1371/journal.pone.0004791  for a given
        set of data. The param() method calculates the log of the data and estimates the abundance data
        from the ranking data using a linear model.
        The coefficients of the linear model are then scaled for future use.
        :param confidence_interval: Numeric. The confidence interval to calculate for the BCD.
        :param _: Internal. Used in the caller() method to ignore unused keyword arguments.
        :return: It returns the given BCDistribution object, but with 4 attributes updated:
                param_data_frame: A data frame with 'pre_numerator' to fit the model and resulting predicted values.
                param_data_matrix: A 3x3 data frame with the parameters of the model with confidence interval
                param_data_model: The RegressionResults object from the OLS model
                param_data_summary: A summary of the OLS model
        """
        rank_data = self.rank_data
        ranking = rank_data.BC_rank
        ranking.name = 'pre_denominator'
        last_place = numpy.max(rank_data.BC_rank)
        pre_numerator = last_place+1-ranking
        pre_numerator.name = 'pre_numerator'
        log_num = numpy.log(pre_numerator)
        log_den = numpy.log(ranking)
        log_abundance = numpy.log(rank_data.abundance)
        xs = pandas.concat(objs=[log_num, log_den], axis=1)
        xs = sm.add_constant(xs)
        lm_model = sm.OLS(log_abundance, xs).fit()
        lm_result = lm_model.params
        params = pandas.Series(lm_result)
        params.name = 'mid'
        param_matrix = lm_model.conf_int(1-confidence_interval)
        param_matrix = param_matrix.join(params)
        param_matrix.iloc[0, :] = numpy.exp(param_matrix.iloc[0, :])
        param_matrix.iloc[2, :] = -param_matrix.iloc[2, :]
        param_matrix.index = ['A', 'b', 'a']
        param_matrix = param_matrix.rename(columns={param_matrix.columns[0]: 'lwr', param_matrix.columns[1]: 'upr'})
        param_matrix = param_matrix.reindex(index=['A', 'a', 'b'], columns=['lwr', 'mid', 'upr'])
        y_preds = numpy.exp(lm_model.get_prediction(xs).summary_frame(alpha=1-confidence_interval))
        y_preds = y_preds[['mean_ci_lower', 'mean', 'mean_ci_upper']]
        y_preds = y_preds.rename(columns={'mean_ci_lower': 'lwr', 'mean': 'mid', 'mean_ci_upper': 'upr'})
        y_preds[y_preds <= 0] = numpy.finfo(float).eps
        param_frame = rank_data.join([pre_numerator, y_preds])
        self.param_data_frame = param_frame
        self.param_data_matrix = param_matrix
        self.param_data_model = lm_model
        self.param_data_summary = lm_model.summary(alpha=1-confidence_interval)
        return self

    def model(self, to_predict=1, **_):
        """
        The model() method has to be called after the param() method has been called.
        Uses the calculated parameters to predict a BCDistribution data point.
        :param to_predict: Number. The rank (x) to predict. It can't be higher than the ranks in the given distribution.
                            Default 1.
        :param _: Internal. Used in the caller() method to ignore unused keyword arguments.
        :return: It returns the given BCDistribution object, but with 1 attribute updated:
                model_data: The predicted abundance (y) value.
        """
        param_matrix = self.param_data_matrix
        numerator = math.pow((max(self.rank_data.BC_rank) + 1 - to_predict),  param_matrix.iloc[2, 1])
        denominator = math.pow(to_predict, param_matrix.iloc[1, 1])
        prediction = param_matrix.iloc[0, 1] * numerator / denominator
        self.model_data = prediction
        return self

    def gof(self, model_extra='MSE',  **_):
        """
        The gof() method calculates a value that describes the goodness of the fit of the predicted to the real data.
        Has to be called after param().
        :param model_extra: Can be one of 'MSE' (Mean Square Error),"S" (Standard error of the Estimate),"R2".
                            Defaults to "MSE".
        :param _: Internal. Used in the caller() method to ignore unused keyword arguments.
        :return: It returns the given BCDistribution object, but with 2 attributes updated:
                gof_data_model: The name of the calculated goodness of fit.
                gof_data_value: The value (number) of the calculated goodness of fit.
        """
        df = self.param_data_frame
        n = max(df.BC_rank)
        t_error = df.mid-df.abundance
        sqerror = numpy.power(t_error, 2)
        ss_res = numpy.sum(sqerror)
        if model_extra == 'MSE':
            value = ss_res/n
        elif model_extra == 'S':
            value = math.sqrt(ss_res/(n-2))
        elif model_extra == 'R2':
            t_mean = numpy.mean(df.abundance)
            t_tot = df.abundance-t_mean
            sqtot = numpy.power(t_tot, 2)
            ss_tot = numpy.sum(sqtot)
            value = 1-ss_res/ss_tot
        else:
            print('ERROR: The type in the gof() function is not supported.')
            value = None
        self.gof_data_model = model_extra
        self.gof_data_value = value
        return self

    def plot(self, obs=True, obs_shape='o', obs_col='C0', obs_size='3', model=True, model_col='C5', model_width=0.5,
             confint=True, confint_col='C1', confint_width=1, confrange=True, confrange_col='C1', gfx_alpha=0.75,
             gfx_title='Rank-Abundance Diagram', gfx_label=True, gfx_label_coords=(83.5/100, 97/100), gfx_xy_trans=None,
             gfx_theme='seaborn-v0_8-pastel', plot_silent=True, gfx_prefix='', **_):
        """
        The plot() method saves a graphical representation of the BCDistribution model.
        It uses matplotlib.pyplot.plot() to produce a png file in the working directory.
        Has to be called after gof().
        :param obs: Logical. Whether to plot the observed abundance data. Defaults to True.
        :param obs_shape: String. The shape of the plotted observed abundance data passed to matplotlib.pyplot.plot().
        :param obs_col: The color for the observations.
        :param obs_size: Numeric. The size for the observations.
        :param model: Logical. Whether to show the models predicted data. Defaults to true.
        :param model_col: Specify a color for the model (midline).
        :param model_width: Numeric. Changes the width of the lines to use for the model (midline).
        :param confint: Logical. Whether to add the confidence interval lines. Defaults to true.
        :param confint_col: Specify a color for the confidence interval lines.
        :param confint_width: Numeric. Changes the width of the confidence interval lines.
        :param confrange: Whether to shade the area in the confidence interval. Defaults to true.
        :param confrange_col: Specify a color to use for the confidence interval shading.
        :param gfx_alpha: Numeric. Modifies all the graphed objects alpha. Default=0.75.
        :param gfx_title: String. Changes the title of the graph.
        :param gfx_label: Logical. Whether to show the parameters used and model_extra info.
        :param gfx_label_coords: Tuple that provides custom x and y values to move the label (must be between 0 and 1).
                                Defaults to (83.5/100, 97/100)
        :param gfx_xy_trans: A list with 2 strings that define the transformations to be applied to the x and y scales.
                                Defaults to ['linear', 'log'].
        :param gfx_theme: Provide a style.contex to pass to pyplot. Defaults to 'seaborn-v0_8-pastel'.
        :param plot_silent: Logical. Whether to show() the produced graph before saving it.
        :param gfx_prefix: A string to prefix the name of the saved file.
        :param _: Internal. Used in the caller() method to ignore unused keyword arguments.
        :return: It returns the given BCDistribution object, but with 1 attribute updated:
                plot_data: The path to the generated graph.
        """
        df = self.param_data_frame
        df = df.reset_index(drop=True)
        df.index = df.index+1
        par = self.param_data_matrix
        plt.figure()
        with plt.style.context(gfx_theme):
            if gfx_xy_trans is None:
                gfx_xy_trans = ['linear', 'log']
            plt.xscale(gfx_xy_trans[0])
            plt.yscale(gfx_xy_trans[1])
            plt.plot(df.upr, color=confint_col, alpha=confint*gfx_alpha, linewidth=confint_width)
            plt.plot(df.mid, color=model_col, alpha=model*gfx_alpha, linewidth=model_width)
            plt.plot(df.lwr, color=confint_col, alpha=confint*gfx_alpha, linewidth=confint_width)
            rev_index = numpy.arange(start=1, stop=max(self.rank_data.BC_rank)+1, step=1)
            plt.fill_between(rev_index, df.upr, df.lwr, alpha=confrange*0.3*gfx_alpha, color=confrange_col)
            plt.plot(df.abundance, obs_shape, color=obs_col, ms=obs_size, alpha=obs*gfx_alpha)
            plt.title(gfx_title)
            plt.xlabel('Rank', fontsize=14)
            plt.ylabel('Abundance', fontsize=14)

            def npbtostr(npb, dec=5):
                t_str = npb.astype('|S'+str(dec+1)).decode('UTF-8')
                return t_str
            if gfx_label:
                plt.annotate((self.gof_data_model + "=" + npbtostr(npb=self.gof_data_value) + "\n" +
                              "A="+npbtostr(npb=par.iloc[0, 1]) + "\n" +
                              "a="+npbtostr(npb=par.iloc[1, 1]) + "\n" +
                              "b="+npbtostr(npb=par.iloc[2, 1])),
                             xy=gfx_label_coords, xycoords='axes fraction', size=8, ha='left',
                             va='top', bbox=dict(boxstyle='round', fc='w'))
        plt.savefig(gfx_prefix+self.rank_data_colname)
        self.plot_data = os.getcwd()+'/'+self.rank_data_colname
        if not plot_silent:
            plt.show()
        plt.close()
        return self

    def report(self, suffix='BCD_', **_):
        """
        The report() method has to be called after the gof() method.
        :param suffix:
        :param _:
        :return: It returns the given BCDistribution object, but with 1 attribute updated:
                report_data: A data frame with the goodness of fit value and paramaters obtained.
        """
        report_frame = {self.rank_data_colname+suffix: [self.gof_data_value, self.param_data_matrix.iloc[0, 1],
                                                        self.param_data_matrix.iloc[1, 1],
                                                        self.param_data_matrix.iloc[2, 1]]}
        report_frame = pandas.DataFrame(report_frame)
        report_frame.index = [self.gof_data_model, 'A', 'a', 'b']
        self.report_data = report_frame
        return self

    def caller(self, dountil='report', args=''):
        """
        Calls rank(), param(), model(), gof(), plot() and report() in that order.
        :param dountil: The string containing the name of the function you wish to reach. Defaults to 'report'
        :param args: A string containing the arguments to be passed to all the functions called. Care should be taken
                    to maintain string arguments. Example: args="model_extra='S'"
        :return: The BCDistribution object which has had the specified methods applied.
        """
        local_environ = {'self': self}
        execution_number = ['rank', 'param', 'model', 'gof', 'plot', 'report'].index(dountil)
        execution_list = ['self = self.rank('+args+')',
                          'self = self.param('+args+')',
                          'self = self.model('+args+')',
                          'self = self.gof('+args+')',
                          'self = self.plot('+args+')',
                          'self = self.report('+args+')']
        for exec_index in range(execution_number+1):
            exec(execution_list[exec_index], local_environ)
        output = local_environ['self']
        return output

    def multiple(self, multiple_args='', col_override=None, json_format=True, json_prefix=''):
        """
        Applies all methods (until report) to all 'n' columns in a data frame and returns a list containing 'n'
        BCDistribution objects. By default, these BCDistributions objects are given as dictionaries that have had all data
        transformed to dictionaries and strings.
        :param multiple_args: String to be passed as 'args' to the caller() method. The rank_column argument is
                            automatically handled.
        :param col_override: A list that contains the indexes of the columns to be analyzed. Must be used when the data
                            frame has a column containing strings or other non-numeric data. Defaults to None.
        :param json_format: Whether to convert the object to dictionaries and strings for JSON serialization, AND
                            save said attributes as a JSON file in the current directory. Defaults to true.
        :param json_prefix: A string prefix to precede the JSON file filename.
        :return: A list containing 'n' BCDistribution objects, or 'n' dictionaries made from the objects attributes.
        """
        multiple_list = list()
        if col_override is None:
            col_override = range(self.df_abundance.shape[1])
        for col_index in col_override:
            index_args = 'column_rank=' + "'" + self.df_abundance.columns[col_index] + "'," + multiple_args
            inside_self = copy.deepcopy(self)
            inside_self = inside_self.caller(dountil='report', args=index_args)
            if json_format:
                inside_self.df_abundance = inside_self.df_abundance.to_dict()
                inside_self.rank_data = inside_self.rank_data.to_dict()
                inside_self.param_data_frame = inside_self.param_data_frame.to_dict()
                inside_self.param_data_matrix = inside_self.param_data_matrix.to_dict()
                inside_self.param_data_model = str(inside_self.param_data_model)
                inside_self.param_data_summary = str(inside_self.param_data_summary)
                inside_self.report_data = inside_self.report_data.to_dict()
                multiple_list.append(vars(inside_self))
            else:
                multiple_list.append(inside_self)
        if json_format:
            with open(json_prefix+'BCDistribution.json', 'w', encoding='utf-8') as f:
                json.dump(multiple_list, f, ensure_ascii=False, indent=4)
        return multiple_list

    @staticmethod
    def many(obj_list, many_args='', many_override=None, many_sep='-', **kwargs):
        """
        The many() method is static. It calls the multiple() method for a list of 'm' objects in the python environment.
        :param obj_list: A list containing the names of the objects to be analyzed. Must be strings.
        :param many_args: Passed to multiple_args in the multiple() method.
        :param many_override: A list containing 'm' lists detailing which columns to analyze for each object.
        :param many_sep: Passed to gfx_prefix and json_prefix arguments combined with the object to analyze name
                        to avoid name collision between saved files in the current directory. Defaults to "-".
        :param kwargs: Can be used to pass the json_format argument to the multiple() method.
        :return: A list containing 'm' lists containing either 'n' BCDistribution objects or dictionaries with their data.
        """
        many_list = []
        if many_override is None:
            for obj in obj_list:
                many_list.append(
                    BCDistribution(globals()[obj]).multiple(multiple_args="gfx_prefix='" +
                                                                          obj + many_sep +
                                                                          "'," + many_args,
                                                            json_prefix=obj+many_sep,
                                                            **kwargs)
                )
        else:
            for obj in obj_list:
                many_list.append(
                    BCDistribution(globals()[obj]).multiple(multiple_args="gfx_prefix='" +
                                                                          obj + many_sep +
                                                                          "'," + many_args,
                                                            col_override=many_override[obj_list.index(obj)],
                                                            json_prefix=obj+many_sep,
                                                            **kwargs)
                )
        return many_list
