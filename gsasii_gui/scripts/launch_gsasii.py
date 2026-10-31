"""Use the pinned upstream GUI and observe its successful project saves."""
from GSASII import GSASIIGUI, GSASIIpath
from project_tracking import ProjectTracker, install_save_tracking


def main():
    original = GSASIIpath.InvokeDebugOpts

    def initialized():
        # Upstream calls this after wx/native setup and GUI-module imports,
        # immediately before creating the frame. Keep its ordinary behavior.
        original()
        from GSASII import GSASIImiscGUI, GSASIIctrlGUI
        install_save_tracking(GSASIImiscGUI, ProjectTracker.environment(),
                              report_error=lambda frame, message: GSASIIctrlGUI.G2MessageBox(
                                  frame, message, 'Galaxy project preservation'))
        print('GSAS-II successful-save tracking active', flush=True)
        GSASIIpath.InvokeDebugOpts = original

    GSASIIpath.InvokeDebugOpts = initialized
    GSASIIGUI.main()


if __name__ == '__main__':
    main()
