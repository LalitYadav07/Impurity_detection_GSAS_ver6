"""Exercise the pinned native Save As and Cancel paths with public test GPXs."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from GSASII import GSASIIpath, GSASIIscriptable
from project_tracking import ProjectTracker, install_save_tracking


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, branch, output = root/'original.gpx', root/'named branch.gpx', root/'galaxy/edited.gpx'
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
    from GSASII import GSASIIdataGUI as gui, GSASIImiscGUI as io
    frame = gui.GSASII(None)
    frame.PackageVersions = {}
    frame.OnFileOpen(None, filename=str(source), askSave=False)
    tracker = ProjectTracker(root, output, root/'Exports')
    tracker.activate(source)
    errors = []
    install_save_tracking(io, tracker, report_error=lambda frame, message: errors.append(message))
    controls_id = gui.GetGPXtreeItemId(frame, frame.root, 'Controls')
    controls = frame.GPXtree.GetItemPyData(controls_id)
    controls['radar_public_save_as_marker'] = 'saved native branch'

    class Dialog:
        def __init__(self, result): self.result = result
        def ShowModal(self): return self.result
        def GetPath(self): return str(branch)
        def GetDirectory(self): return str(root)
        def GetWildcard(self): return 'GSAS-II project file (*.gpx)|*.gpx'
        def GetFilterIndex(self): return 0
        def Destroy(self): pass

    try:
        with patch.object(wx, 'FileDialog', return_value=Dialog(wx.ID_OK)):
            assert frame.OnFileSaveas(None) is True
        assert not errors, errors
        assert json.loads(tracker.active.read_text())['path'] == str(branch)
        assert output.read_bytes() == branch.read_bytes() != source.read_bytes()
        readback = GSASIIscriptable.G2Project(gpxfile=str(output))
        assert readback.data['Controls']['data']['radar_public_save_as_marker'] == 'saved native branch'
        before = output.read_bytes()
        pointer = tracker.active.read_bytes()
        with patch.object(wx, 'FileDialog', return_value=Dialog(wx.ID_CANCEL)):
            assert frame.OnFileSaveas(None) is False
        assert tracker.active.read_bytes() == pointer and output.read_bytes() == before
        index = json.loads(tracker.index.read_text())
        assert {row['name'] for row in index['projects'].values()} == {source.name, branch.name}
        for row in index['projects'].values():
            GSASIIscriptable.G2Project(gpxfile=str(tracker.index.parent/row['file']))
        print('Native Save As, cancelled Save As, edited GPX readback and named branch preservation passed')
    finally:
        frame.Destroy()
        app.Destroy()


if __name__ == '__main__':
    main()
