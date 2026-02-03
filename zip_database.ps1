$sourceDir = "data\database_aug"
$zipFile = "database_aug.zip"

echo "📦 Zipping essential database files..."

# Define essential items 
$include = @(
    "highsymm_metadata.json",
    "catalog_deduplicated.csv",
    "mp_experimental_stable.csv",
    "profiles64"
)

# Create a temporary folder for staging
$stageDir = "temp_stage_db"
if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
New-Item -ItemType Directory -Path "$stageDir\database_aug" | Out-Null

# Copy essential files
foreach ($item in $include) {
    echo "   + Copying $item"
    Copy-Item -Recurse -Path "$sourceDir\$item" -Destination "$stageDir\database_aug"
}

# Zip it
echo "📚 Creating archive: $zipFile"
if (Test-Path $zipFile) { Remove-Item -Force $zipFile }
Compress-Archive -Path "$stageDir\database_aug\*" -DestinationPath $zipFile

# Cleanup
Remove-Item -Recurse -Force $stageDir

echo "✅ Done! Upload '$zipFile' to Google Drive or GitHub Releases."
