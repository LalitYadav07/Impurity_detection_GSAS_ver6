"""Keep the native plot notebook alive while its project is open.

The upstream data-tree handlers retain G2plotNB. Some Linux window managers
still offer Close even though GSAS-II removes the plot frame's close-box flag.
Destroying that frame leaves every subsequent tree selection with a deleted
AuiNotebook. Treat an ordinary plot-window close as hide, and show the same
notebook again when a native plotting action requests a tab.
"""
from functools import wraps


def install_plot_window_lifecycle(frame_type, notebook_type):
    import wx

    if getattr(frame_type, '_radar_plot_lifecycle_installed', False):
        return
    original_init = frame_type.__init__
    original_find = notebook_type.FindPlotTab

    @wraps(original_init)
    def initialize(frame, *args, **kwargs):
        original_init(frame, *args, **kwargs)
        plot = frame.plotFrame

        def close_plot(event):
            if event.CanVeto() and not frame.IsBeingDeleted():
                event.Veto()
                plot.Hide()
                frame.Raise()
            else:
                # Forced shutdown must remain able to destroy the desktop.
                event.Skip()

        plot.Bind(wx.EVT_CLOSE, close_plot)

    @wraps(original_find)
    def find_plot(notebook, *args, **kwargs):
        result = original_find(notebook, *args, **kwargs)
        plot = notebook.GetTopLevelParent()
        if not plot.IsShown():
            plot.Show()
            plot.Raise()
        return result

    frame_type.__init__ = initialize
    notebook_type.FindPlotTab = find_plot
    frame_type._radar_plot_lifecycle_installed = True
