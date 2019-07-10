import plotly.graph_objs as go
import warnings as w


# Traces
def generate_trace_data(df,
                        name,
                        kind='box',
                        custom=True,
                        y_field=None,
                        x_field=None):
    """
    Generate trace data

    In essence, a 'trace' is a set of datapoints being plotted
    on your graph.


    :param df:          DataFrame of correct form, e.g.:
                                          | X-Data | Y-Data
                                ID | Name |
                                01 | test |     03  |   05

                        This function assumes a DataFrame with a (multi)-index
                        called name, selects a specific name and identifies the
                        2 columns (excluding indices!) of the DataFrame as x
                        and y.
    :param name:        Name, (multi)-index present in the DataFrame used to
                        filter the data that should be included in our plot.
    :param kind:        Kind of plot that this trace data will ultimately be,
                        this is only relevant for the customdata field, which
                        in turn is only relevant when we are looking to a
                        boxplot. In a sense, this is a double-check.
    :param custom:      Flag that indicates whether or not we want customdata
                        embedded in our trace-data
    :return:
    """

    # Build trace data dict
    tmp = df.xs(name, level='name') # Subset of original df

    if kind in {'bar', 'barplot'}:
        x_default = 'dataset'
        y_default = 'time (s)'
    elif kind in {'box', 'boxplot'}:
        x_default = 'dataset'
        y_default = 'score'
    elif kind in {'line', 'lineplot', 'scatter', 'scatterplot'}:
        x_default = 'perc_miss'
        y_default = 'rank'

        if x_default in tmp.columns.values:
            tmp = tmp.sort_values(by=x_default) # Otherwise the lines look like shit
    else:
        w.warn("Did not recognize kind: {}."
               "Assuming default option of barplot instead".format(kind))
        x_default = 'dataset'
        y_default = 'time (s)'

    if x_field is None:
        x_field = x_default
    if y_field is None:
        y_field = y_default

    trace_data = {'x':          tmp[x_field].values,
                  'y':          tmp[y_field].values,
                  'visible':    True,
                  'name':       name}

    # We add some customdata to the boxplot
    if custom and (kind in {'box', 'boxplot'}):
        trace_data = {'customdata': tmp.index.values,
                      **trace_data}

    return trace_data


def generate_trace_layout(kind=None, show_data=False):
    """
    Generate the layout of a given trace.

    This also (counter-intuitively) involves the actual kind of
    plot in which the trace data will be represented, e.g., boxplot, barplot...

    :param kind:    The form in which the trace data will ultimately be represented
                    I.e.; boxplot or barplot.
                    Default: barplot
    :return:
    """

    if kind in {'bar', 'barplot'}:
        layout = {}
    elif kind in {'box', 'boxplot'}:
        show_data_layout = {'boxpoints':  'all',
                            'jitter':     1,
                            'pointpos':   -2,
                            'marker':     {'symbol':  'y-up-open',
                                           'size':    2}}

        layout = {'boxmean':        'sd',
                  'whiskerwidth':   0.25}

        if show_data:
            layout = {**show_data_layout,
                      **layout}
    elif kind in {'line', 'lineplot'}:

        layout = {}
    elif kind in {'scatter', 'scatterplot'}:
        layout = {'mode': 'markers'}
    else:
        w.warn("Did not recognize kind: {}. Assuming default option of barplot"
               "instead".format(kind))
        layout = generate_trace_layout(kind='bar')

    return layout


def generate_trace(trace_data, trace_layout, kind=None):
    """
    Join trace data and trace layout.

    This results in an object ready to be embedded in a Plotly graph.

    :param trace_data:      Trace-data
    :param trace_layout:    Trace-Layout
    :param kind:            The form in which the trace data will ultimately
                            be represented, i.e.:
                                - Boxplot
                                - Barplot

                                Default: Barplot
    :return:
    """

    if kind in {'bar', 'barplot'}:
        trace = go.Bar(**trace_data, **trace_layout)
    elif kind in {'box', 'boxplot'}:
        trace = go.Box(**trace_data, **trace_layout)
    elif kind in {'line', 'lineplot', 'scatter', 'scatterplot'}:
        trace = go.Scatter(**trace_data, **trace_layout)
    else:
        w.warn("Did not recognize kind: {}. Assuming default option of barplot"
               "instead".format(kind))
        trace = go.Bar(**trace_data, **trace_layout)

    return trace


# Graphs
def generate_graph_traces(df,
                          kind=None,
                          show_data=False,
                          custom=True,
                          y_field=None,
                          x_field=None):
    """
    Generate graph traces, i.e.; traces that make up the graph.

    A trace is essentially a bunch of data that is somehow related together,
    plotted in a consistent manner on the graph. For example, a single
    line on a line graph would be a trace. This makes it very easy to plot
    multiple lines on the same graph without too much hassle.

    For other kinds of graph the definition can slightly change, but the overall
    idea remains, it is just convenient to not always have to talk about new
    'plots' that are plotted on the same `canvas' (cf. matplotlib).

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame that contains the information we wish to plot.
        This DataFrame should adhere to a specific structure, i.e.:
            A DataFrame with a (multi)-index called `name'. For every
            *distinct* name in this index, we generate a trace from the
            two actual (i.e. non-index) columns of the DataFrame, which
            are identified as the x and y components of the corresponding
            trace.
        E.g.:
                            | X-Data | Y-Data
                ID   | Name |
                01   | test |     03 |   05

    kind: str, {Boxplot, Barplot, Lineplot}, default: Barplot
        The kind of graph that we are generating, i.e.:
    show_data: bool
        Whether or not to show the raw data in the scatterplot
    custom
    y_field
    x_field

    Returns
    -------

    """

    traces = []

    df.sort_index(level='name', inplace=True)

    for name in df.index.get_level_values(level='name').unique().values:        # This requires an index entry named `name' (Slight hardcoding)
        trace_data = generate_trace_data(df,
                                         name,
                                         kind=kind,
                                         custom=custom,
                                         y_field=y_field,
                                         x_field=x_field)
        trace_layout = generate_trace_layout(kind=kind, show_data=show_data)

        trace = generate_trace(trace_data, trace_layout, kind=kind)

        traces.append(trace)

    return traces


def generate_graph_layout(kind=None, x_title=None, y_title=None):
    """
    Generate default graph layout

    :param kind:            The kind of graph that we are generating, i.e.:
                                - Boxplot
                                - Barplot

                                Default: Barplot
    :return:                Dictionary with all the relevant parameters.
    """

    if kind in {'bar', 'barplot'}:
        if y_title is None:
            y_title = 'Induction time (s)'
        if x_title is None:
            x_title = 'Dataset'

        barplot_layout_parameters = {'barmode':     'group',
                                     'height':      1000,
                                     'width':       1500,
                                     'yaxis':       {'autorange':   True,
                                                     'title':       y_title,
                                                     'titlefont':   {'size': 45}},
                                     'xaxis':       {'title':       x_title,
                                                     'titlefont':   {'size': 45}},
                                     'title':       y_title,
                                     'titlefont':   {'size': 35}}

        layout = go.Layout(**barplot_layout_parameters)
    elif kind in {'box', 'boxplot'}:
        if y_title is None:
            y_title = 'Macro-F1'
        if x_title is None:
            x_title = 'Dataset'

        boxplot_layout_parameters = {'boxmode':     'group',
                                     'height':      1000,
                                     'width':       1500,
                                     'yaxis':       {#'range':       [-1, 1],
                                                     'zeroline':    True,
                                                     'title':       y_title,
                                                     'titlefont':   {'size': 45}},
                                     'xaxis':       {'title':       x_title,
                                                     'titlefont':   {'size': 45}},
                                     'boxgap':      0.3,
                                     'boxgroupgap': 0.6,
                                     'title':       'Predictive Performance',
                                     'titlefont':   {'size': 35}}

        layout = go.Layout(**boxplot_layout_parameters)
    elif kind in {'line', 'lineplot'}:
        if y_title is None:
            y_title = 'Average Rank'
        if x_title is None:
            x_title = 'Perc. Missing (%)'

        lineplot_layout_parameters = {'height':     1000,
                                      'width':      1500,
                                      'yaxis':      {'autorange':   True,
                                                     'title':       y_title,
                                                     'titlefont':   {'size': 45}},
                                      'xaxis':      {'title':       x_title,
                                                     'titlefont':   {'size': 45}},
                                      'title':       y_title,
                                      'titlefont':  {'size': 35},
                                      'legend':     {'font':        {'size':  10},
                                                     'borderwidth': 1,
                                                     'x':           0.9,
                                                     'y':           1.5}
                                      }

        layout = go.Layout(**lineplot_layout_parameters)
    elif kind in {'scatter', 'scatterplot'}:
        if y_title is None:
            y_title = 'Average Inference Time'
        if x_title is None:
            x_title = 'Dataset'

        scatterplot_layout_parameters = {'height':     1000,
                                         'width':      1500,
                                         'yaxis':      {'autorange':   True,
                                                        'title':       y_title,
                                                        'titlefont':   {'size': 45}},
                                         'xaxis':      {'title':       x_title,
                                                        'titlefont':   {'size': 45}},
                                         'title':       y_title,
                                         'titlefont':  {'size': 35},
                                         'legend':     {'font':        {'size':  10},
                                                        'borderwidth': 1,
                                                        'x':           0.9,
                                                        'y':           1.5}
                                         }

        layout = go.Layout(**scatterplot_layout_parameters)

    else:
        w.warn("Did not recognize kind: {}. Assuming default option of barplot"
               "instead".format(kind))
        layout = generate_graph_layout(kind='bar')

    return layout


def generate_graph(df,
                   kind=None,
                   show_data=False,
                   custom=True,
                   y_field=None,
                   x_field=None,
                   y_title=None,
                   x_title=None):
    """
    Generate a dictionary that represents an entire graph.

    :param df:      DataFrame that contains the information we wish to plot.
                    This DataFrame should adhere to a specific structure, i.e.;

                    A DataFrame with a (multi)-index called name. For every
                    distinct name in this index, we generate a trace from the
                    two actual (i.e. non-index) columns of the DataFrame, which
                    are identified as the x and y components of the corresponding
                    trace.
                    E.g.;
                                            | X-Data | Y-Data
                                ID   | Name |
                                01   | test |     03 |   05

    :param kind:    The kind of graph that we are generating, i.e.:
                                - Boxplot
                                - Barplot

                                Default: Barplot
    :return:
    """

    traces = generate_graph_traces(df,
                                   kind=kind,
                                   show_data=show_data,
                                   custom=custom,
                                   y_field=y_field,
                                   x_field=x_field)
    layout = generate_graph_layout(kind=kind, y_title=y_title, x_title=x_title)

    graph = {'data':    traces,
             'layout':  layout}

    return graph
