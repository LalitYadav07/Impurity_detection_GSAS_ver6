"""Native wx regression for plot close, tree selection, reopen and exit."""
from pathlib import Path
import sys

from GSASII import GSASIIpath, GSASIIscriptable
from plot_window_lifecycle import install_plot_window_lifecycle


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = '--without-guard' in sys.argv
    sys.argv = sys.argv[:1]
    source = root / 'public.gpx'
    GSASIIscriptable.G2Project(newgpx=str(source)).save()
    GSASIIpath.SetBinaryPath(True)
    import wx
    import wx.grid
    import wx.aui
    import wx.lib.scrolledpanel
    import wx.lib.mixins.listctrl
    import wx.richtext
    import wx.lib.filebrowsebutton
    app = wx.App(False)
    from GSASII import GSASIIdataGUI as gui, GSASIIplot
    if not baseline:
        install_plot_window_lifecycle(gui.GSASII, GSASIIplot.G2PlotNoteBook)
        install_plot_window_lifecycle(gui.GSASII, GSASIIplot.G2PlotNoteBook)
    frame = gui.GSASII(None)
    frame.PackageVersions = {}
    frame.OnFileOpen(None, filename=str(source), askSave=False)
    try:
        notebook = frame.G2plotNB
        new, index, page, axes, limits = notebook.FindPlotTab('Public control', 'mpl')
        axes.plot([1, 2], [3, 4])
        wx.Yield()
        frame.plotFrame.Close()
        wx.Yield()
        if baseline:
            try:
                notebook.nb.GetSelection()
            except RuntimeError as error:
                assert 'deleted' in str(error)
                print('Reproduced deleted AuiNotebook after plot-window close')
                return
            raise AssertionError('The unguarded plot window unexpectedly survived Close')
        assert not frame.plotFrame.IsShown()
        assert notebook.nb.GetPageCount() == 1
        controls = gui.GetGPXtreeItemId(frame, frame.root, 'Controls')
        event = wx.TreeEvent(wx.wxEVT_TREE_SEL_CHANGED, frame.GPXtree, controls)
        frame.OnDataTreeSelChanged(event)
        wx.Yield()
        reopened = notebook.FindPlotTab('Public control', 'mpl', newImage=False)
        assert reopened[0] is False and reopened[2] is page
        assert frame.plotFrame.IsShown()
        assert len(reopened[3].lines) == 1
        frame.plotFrame.Close(force=True)
        wx.Yield()
        assert not frame.plotFrame
        print('Plot close, tree navigation, preserved plot reopen and forced shutdown passed')
    finally:
        if frame.plotFrame:
            frame.plotFrame.Destroy()
        frame.Destroy()
        app.Destroy()


if __name__ == '__main__':
    main()
