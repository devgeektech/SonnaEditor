# Saha Lightroom Bridge

This Lightroom Classic plugin imports `sonna_lightroom_edits.lua`, the bridge
package written beside `sonna_predictions.json` after a Saha processing run.

Install it through Lightroom Classic:

1. File -> Plug-in Manager.
2. Add this `SahaBridge.lrplugin` folder.
3. After a catalog processing run, choose Library -> Plug-in Extras -> Import
   Saha Edits and select the generated `sonna_lightroom_edits.lua`.

The Python app still writes XMP sidecars next to the RAWs. This plugin applies
the same settings through Lightroom's catalog write API so an already-open
catalog can show the edits without direct SQLite writes.