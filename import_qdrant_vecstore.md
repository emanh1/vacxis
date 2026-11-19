# Restore Qdrant Collection (VacxiS_Capstone_FA26)

Run one of the following commands from the project root (ensure local Qdrant is running on port 6333):

### Linux / macOS:
curl -X POST "http://localhost:6333/collections/VacxiS_Capstone_FA26/snapshots/upload?priority=snapshot" \
  -F "snapshot=@chunks_storage/VacxiS_Capstone_FA26.snapshot"

### Windows (PowerShell):
curl.exe -X POST "http://localhost:6333/collections/VacxiS_Capstone_FA26/snapshots/upload?priority=snapshot" `
  -F "snapshot=@chunks_storage/VacxiS_Capstone_FA26.snapshot"

### Windows (CMD):
curl -X POST "http://localhost:6333/collections/VacxiS_Capstone_FA26/snapshots/upload?priority=snapshot" ^
  -F "snapshot=@chunks_storage\VacxiS_Capstone_FA26.snapshot"

### Verify:
- Linux / macOS:
  curl -s http://localhost:6333/collections/VacxiS_Capstone_FA26
- Windows (PowerShell / CMD):
  curl.exe -s http://localhost:6333/collections/VacxiS_Capstone_FA26

Check collection health, vector configuration and actual point count after restoration. The checked-in Parquet corpus currently contains 1,744 chunks across 75 document IDs.

[PLACEHOLDER: verify the snapshot's actual point count and chunk IDs against the Parquet corpus and record the matching revision before using it for evaluation.]
