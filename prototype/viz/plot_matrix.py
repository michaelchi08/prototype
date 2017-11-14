import numpy as np
import matplotlib.pylab as plt


class PlotMatrix:
    """Plot matrix

    Attributes
    ----------
    show_ticks : bool
        Show ticks
    show_values : bool
        Show values in plot
    show : bool
        Show plot
    labels : :obj`list` of :obj`str`
        Plot labels

    rows : int
        Number of rows
    cols : int
        Number of columns
    fig : matplotlib.figure.Figure
        Figure
    plt_ax : matplotlib.axes.Axes
        Plot axis
    cov_ax : matplotlib.axes.Axes
        Covariance axis

    color_bar : matplotlib
        Color bar

    Parameters
    ----------
    show_ticks : bool
        Show ticks
    show_values : bool
        Show values in plot
    show : bool
        Show plot
    labels : :obj`list` of :obj`str`
        Plot labels

    """

    def __init__(self, data, **kwargs):
        # Settings
        self.show_ticks = kwargs.get("show_ticks", False)
        self.show_values = kwargs.get("show_values", False)
        self.show = kwargs.get("show", False)
        self.labels = kwargs.get("labels", None)

        # Setup plot
        self.rows, self.cols = data.shape
        self.fig = plt.figure()
        self.plt_ax = self.fig.add_subplot(111)
        self.cov_ax = self.plt_ax.matshow(np.array(data))

        # Covariance matrix labels
        self.label_values = self._add_data_labels(data)
        self._add_axis_labels(data)

        # Color bar
        self.color_bar = self.fig.colorbar(self.cov_ax)

        # Show plot
        if self.show:
            plt.show(block=False)

    def _add_data_labels(self, data):
        """Add matrix values into the plot

        Parameters
        ----------
        data: np.array
            Covariance matrix data

        Returns
        -------
        label_values : matplotlib.text.Text
            List of matplotlib text labels, each representing a cell in the
            plot

        """
        if self.show_values is False:
            return

        m, n = data.shape
        label_values = []
        for i in range(m):
            for j in range(n):
                c = data[i][j]
                txt = self.plt_ax.text(i, j,
                                       str(round(c, 2)),
                                       va='center',
                                       ha='center')
                label_values.append(txt)

        return label_values

    def _add_axis_labels(self, data):
        """Add matrix axis labels

        Parameters
        ----------
        data: np.array
            Covariance matrix data

        """
        if self.show_ticks is False:
            return

        m, n = data.shape
        self.plt_ax.set_xticks(np.arange(n + 1))
        self.plt_ax.set_yticks(np.arange(m + 1))

        if self.labels is not None:
            self.plt_ax.set_xticklabels(self.labels)
            self.plt_ax.set_yticklabels(self.labels)

    def _update_data_labels(self, data):
        """Update data values in plot

        Parameters
        ----------
        data: np.array
            Covariance matrix data

        """
        if self.show_values is False:
            return

        index = 0
        m, n = data.shape

        for i in range(m):
            for j in range(n):
                txt = self.label_values[index]
                txt.set_text(str(round(data[i][j], 0)))
                index += 1

    def _update_color_bar(self, data):
        """Update color bar

        Parameters
        ----------
        data: np.array
            Covariance matrix data

        """
        color_bar_ticks = np.linspace(np.min(data), np.max(data),
                                      num=11, endpoint=True)
        self.color_bar.set_ticks(color_bar_ticks)
        self.color_bar.set_clim(vmin=np.min(data), vmax=np.max(data))
        self.color_bar.draw_all()

    def update(self, data):
        """Update the plot

        Parameters
        ----------
        data: np.array
            Covariance matrix data

        """
        if data.shape[0] > self.rows or data.shape[1] > self.cols:
            self.plt_ax.clear()
            self.cov_ax = self.plt_ax.matshow(np.array(data))
            self._add_data_labels(data)
            self._add_axis_labels(data)

        else:
            self.cov_ax.set_data(np.array(data))
            self._update_data_labels(data)
            self._update_color_bar(data)

        # Update plot
        self.fig.canvas.draw()
