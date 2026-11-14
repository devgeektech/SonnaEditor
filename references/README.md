# UI & Workflow Reference Images

This folder stores reference screenshots and UX patterns from Imagen used to guide the architecture, UI design, and workflow implementation of Sonna Editor (Saha App).

You can drop additional reference images directly into this folder at any time.

---

## Reference Index (40 Screenshots)

### 1. Onboarding & Core Views
| File | Description | Core UX Pattern |
|---|---|---|
| `01_imagen_home_welcome.png` | Home screen welcome view | 3-step onboarding banner (Add profile → Create project → Edit) |
| `02_imagen_ai_profiles_empty.png` | AI Profiles tab (empty state) | Empty state call-to-action to create first profile |
| `03_imagen_projects_empty.png` | Projects tab (empty state) | Clean starting point for shoot-based project workflow |

### 2. Profile Selection & Qualification Wizard
| File | Description | Core UX Pattern |
|---|---|---|
| `04_imagen_choose_profile_type_modal.png` | Profile selection modal | Split between **Lite Personal AI Profile** (preset + survey) and **Personal AI Profile** (catalog training) |
| `05_imagen_profile_wizard_q1_photo_count.png` | Profile wizard: Question 1 | "Do you have at least 2,000 photos edited in Lightroom Classic?" |
| `06_imagen_profile_wizard_q2_consistent_style.png` | Profile wizard: Question 2 | Style consistency across different lighting conditions |
| `07_imagen_profile_wizard_q3_easy_access.png` | Profile wizard: Question 3 | RAW/catalog accessibility check |

### 3. Mode B (Lite Profile) Setup
| File | Description | Core UX Pattern |
|---|---|---|
| `08_imagen_lite_profile_step1_name_and_upload.png` | Lite Profile setup (empty) | Profile naming input + preset XMP drag-and-drop zone |
| `09_imagen_lite_profile_step1_preset_dropdown.png` | Lite Profile preset dropdown | Searchable dropdown for Lightroom presets |
| `10_imagen_lite_profile_step1_preset_selected.png` | Lite Profile preset card | Uploaded preset tile with file size, remove button, and Next CTA |

### 4. Mode B (Lite Profile) Style Survey
| File | Description | Core UX Pattern |
|---|---|---|
| `11_imagen_lite_survey_exposure_bride_lake.jpg` | Style survey: Exposure (Landscape) | 5-step visual comparison slider (Darker to Brighter) |
| `12_imagen_lite_survey_exposure_child_darkest_selected.jpg` | Style survey: Exposure (Portrait) | Selected variation highlight (active selection state) |
| `13_imagen_lite_survey_exposure_bride_garden.jpg` | Style survey: Exposure (Garden) | Visual survey applied across warm green foliage |
| `14_imagen_lite_survey_temperature_bride_lake.jpg` | Style survey: Temperature (Landscape) | Cooler (blue) to Warmer (yellow) color gradient slider |
| `15_imagen_lite_survey_temperature_child.jpg` | Style survey: Temperature (Portrait) | Multi-scene temperature visual variations |
| `16_imagen_lite_survey_temperature_bride_garden.jpg` | Style survey: Temperature (Garden) | Foliage and skin tones under warm vs cool lighting |
| `17_imagen_lite_survey_tint_bride_lake.jpg` | Style survey: Tint (Landscape) | Green to Magenta color gradient slider |
| `18_imagen_lite_survey_tint_child.jpg` | Style survey: Tint (Portrait) | Skin tone tint visual variations |
| `19_imagen_lite_survey_tint_bride_garden_build_cta.jpg` | Style survey: Tint (Final Question) | Tint selection with final "Build profile" action button |

### 5. Profile Generation & Ready States
| File | Description | Core UX Pattern |
|---|---|---|
| `20_imagen_building_profile_modal.jpg` | Building AI Profile progress modal | Animated fingerprint graphic with upload % status |
| `21_imagen_profile_ready_modal.jpg` | Profile ready modal | Success confirmation with "Edit with this profile" CTA |

### 6. Project Ingestion & Workflow Sources
| File | Description | Core UX Pattern |
|---|---|---|
| `22_imagen_project_source_lightroom_catalog.png` | Choose source: Lightroom Classic | Catalog selector (recent catalogs dropdown, browse, drag & drop) |
| `23_imagen_project_source_folder.png` | Choose source: Folder | Folder picker supporting JPEG, TIFF, and RAW formats |

### 7. Profile Management & Tuning Overrides
| File | Description | Core UX Pattern |
|---|---|---|
| `24_imagen_profile_list_view.png` | AI Profiles list table | Columns: Profile name, Number of photos, Date, Status, "Use to edit" |
| `25_imagen_profile_update_settings_modal.png` | Update profile parameters modal | Left: 5 RAW preview dropzone. Right: Basic WB, Tone, and Presence adjustments |
| `26_imagen_profile_context_menu_options.png` | Profile context menu | View & modify, Training history, Update profile, Duplicate, Rename, Share, Delete |

### 8. Mode A (Personal AI Profile) Training Flow
| File | Description | Core UX Pattern |
|---|---|---|
| `27_imagen_full_profile_wizard_guidelines.jpg` | Training guidelines | Recommendations for environment variety and style consistency |
| `28_imagen_full_profile_wizard_software_selection.jpg` | Software selection | Lightroom Classic, Lightroom CC, Photoshop ACR, Bridge ACR |
| `29_imagen_full_profile_wizard_format_raw_or_jpeg.jpg` | File format selection | RAW vs JPEG\TIFF training targets |
| `30_imagen_full_profile_wizard_color_or_bw.jpg` | Color profile type | Color vs Black & White target profile |

### 9. Mode A Profile Creation & Catalog Ingestion
| File | Description | Core UX Pattern |
|---|---|---|
| `31_imagen_create_profile_naming.jpg` | Create profile naming modal | Initial profile naming input with step indicator for Personal AI Profile |
| `32_imagen_photography_type_selection.png` | Photography type selection grid | 9-genre selection grid (+ Other) to customize AI evaluation and model defaults |
| `33_imagen_train_profile_source_catalog.png` | Train profile: Lightroom catalog source | Catalog picker with recent catalog dropdown, file browser, and drag-and-drop zone |
| `34_imagen_train_profile_source_folder_raw_formats.png` | Train profile: Folder source with RAW support | Folder picker displaying tooltip of all accepted RAW formats (.cr2, .arw, .dng, .nef, .cr3, etc.) |
| `35_imagen_train_profile_catalog_browse.png` | Train profile: Browse catalog view | Alternative view for browsing local catalog files |

### 10. Project Creation & Advanced AI Tools
| File | Description | Core UX Pattern |
|---|---|---|
| `36_imagen_project_creation_basic_info.png` | New project: Basic info step | Shoot path, photo count validation (e.g. 21 photos), profile selection, and genre assignment |
| `37_imagen_project_ai_tools_denoise_flyout.png` | AI tools: Denoise configuration flyout | Intelligent tool toggles with contextual parameter flyouts (Automatic per photo vs ISO-gated) |
| `38_imagen_denoise_settings_side_by_side.jpg` | Denoise settings modal | 4-tier strength selection (Subtle, Moderate, Strong, Stronger) with side-by-side preview and ISO thresholding |
| `39_imagen_project_dashboard_processing_card.png` | Projects dashboard: Active processing card | Project card with dynamic status chip (Processing), photo count, and background upload progress bar |
| `40_imagen_project_dashboard_choose_source_modal.png` | Project dashboard: Source selection modal | Modal prompt to select between catalog or folder ingestion when initiating a new project |

---

## Adding More References

1. Save new reference screenshots or assets into `references/`.
2. Follow sequential descriptive naming (e.g. `41_culling_view.png`, `42_export_xmp_dialog.png`).
3. We can reference these directly when designing or updating UI components, survey interactions, and onboarding flows.
