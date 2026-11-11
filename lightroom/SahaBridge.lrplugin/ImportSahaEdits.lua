local LrApplication = import "LrApplication"
local LrDialogs = import "LrDialogs"
local LrTasks = import "LrTasks"

local function indexPhotosByPath(catalog)
  local pathToPhoto = {}
  for _, photo in ipairs(catalog:getAllPhotos()) do
    local path = photo:getRawMetadata("path")
    if path and not pathToPhoto[path] then
      pathToPhoto[path] = photo
    end
  end
  return pathToPhoto
end

LrTasks.startAsyncTask(function()
  local result = LrDialogs.runOpenPanel({
    title = "Choose Saha edit package",
    canChooseFiles = true,
    canChooseDirectories = false,
    allowsMultipleSelection = false,
    fileTypes = { "lua" },
  })
  if not result or not result[1] then
    return
  end

  local ok, package = pcall(dofile, result[1])
  if not ok or type(package) ~= "table" or type(package.photos) ~= "table" then
    LrDialogs.message("Saha Lightroom Bridge", "Could not read this edit package.", "critical")
    return
  end

  local catalog = LrApplication.activeCatalog()
  local pathToPhoto = indexPhotosByPath(catalog)
  local applied = 0
  local missing = 0
  local errors = 0

  catalog:withWriteAccessDo("Apply Saha edits", function()
    for _, edit in ipairs(package.photos) do
      local photo = pathToPhoto[edit.raw_path]
      if photo and type(edit.settings) == "table" then
        local okApply = pcall(function()
          photo:applyDevelopSettings(edit.settings)
        end)
        if okApply then
          applied = applied + 1
        else
          errors = errors + 1
        end
      else
        missing = missing + 1
      end
    end
  end)

  LrDialogs.message(
    "Saha Lightroom Bridge",
    string.format("Applied %d edits. Missing: %d. Errors: %d.", applied, missing, errors),
    "info"
  )
end)
