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


# Restore LightRAG Qdrant Collections (workspace: `vacxis`)

Ensure local Qdrant is running on port 6333, then upload all 3 `baai/bge-m3` snapshots from `lightrag_storage/snapshots/`:

### Linux / macOS:
```bash
for col in \
  lightrag_vdb_chunks_baai_bge_m3_1024d \
  lightrag_vdb_entities_baai_bge_m3_1024d \
  lightrag_vdb_relationships_baai_bge_m3_1024d; do
  curl -X POST "http://localhost:6333/collections/${col}/snapshots/upload?priority=snapshot" \
    -F "snapshot=@lightrag_storage/snapshots/${col}.snapshot"
done
```

### Windows (PowerShell):
```powershell
$cols = @(
  "lightrag_vdb_chunks_baai_bge_m3_1024d",
  "lightrag_vdb_entities_baai_bge_m3_1024d",
  "lightrag_vdb_relationships_baai_bge_m3_1024d"
)
foreach ($col in $cols) {
  curl.exe -X POST "http://localhost:6333/collections/$col/snapshots/upload?priority=snapshot" `
    -F "snapshot=@lightrag_storage/snapshots/$col.snapshot"
}
```

### Windows (CMD):
```cmd
curl -X POST "http://localhost:6333/collections/lightrag_vdb_chunks_baai_bge_m3_1024d/snapshots/upload?priority=snapshot" ^
  -F "snapshot=@lightrag_storage\snapshots\lightrag_vdb_chunks_baai_bge_m3_1024d.snapshot"
curl -X POST "http://localhost:6333/collections/lightrag_vdb_entities_baai_bge_m3_1024d/snapshots/upload?priority=snapshot" ^
  -F "snapshot=@lightrag_storage\snapshots\lightrag_vdb_entities_baai_bge_m3_1024d.snapshot"
curl -X POST "http://localhost:6333/collections/lightrag_vdb_relationships_baai_bge_m3_1024d/snapshots/upload?priority=snapshot" ^
  -F "snapshot=@lightrag_storage\snapshots\lightrag_vdb_relationships_baai_bge_m3_1024d.snapshot"
```

### Verify Qdrant LightRAG Collections:
```bash
for col in \
  lightrag_vdb_chunks_baai_bge_m3_1024d \
  lightrag_vdb_entities_baai_bge_m3_1024d \
  lightrag_vdb_relationships_baai_bge_m3_1024d; do
  echo "$col:"
  curl -s "http://localhost:6333/collections/${col}"
  echo ""
done
```

Counts recorded in the original restore instructions (verify against the restored service):
- `lightrag_vdb_chunks_baai_bge_m3_1024d`: `points_count=1492`
- `lightrag_vdb_entities_baai_bge_m3_1024d`: `points_count=3572`
- `lightrag_vdb_relationships_baai_bge_m3_1024d`: `points_count=1830`


# Restore LightRAG Neo4j Knowledge Graph (workspace: `vacxis`)

Ensure Neo4j is running and `.env` defines `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` (defaults: `bolt://localhost:7687`, `neo4j` / `neo4j`, database `neo4j`).

Snapshot file: `lightrag_storage/snapshots/neo4j_vacxis.json`

### Import (Linux / macOS / Windows PowerShell via `uv`):
```bash
uv run python -c '
import json, os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(".env")
data = json.loads(Path("lightrag_storage/snapshots/neo4j_vacxis.json").read_text(encoding="utf-8"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
)
db = os.getenv("NEO4J_DATABASE", "neo4j")

with driver.session(database=db) as s:
    s.run("MATCH (n:vacxis) DETACH DELETE n")
    s.run("CREATE INDEX IF NOT EXISTS FOR (n:vacxis) ON (n.entity_id)")
    s.run("CREATE FULLTEXT INDEX entity_id_fulltext_idx_vacxis IF NOT EXISTS FOR (n:vacxis) ON EACH [n.entity_id]")

    nodes = [n["properties"] for n in data["nodes"]]
    for i in range(0, len(nodes), 500):
        s.run(
            "UNWIND $batch AS props CREATE (n:vacxis) SET n = props",
            batch=nodes[i:i+500],
        )

    rels = data["relationships"]
    for i in range(0, len(rels), 500):
        s.run(
            "UNWIND $batch AS r "
            "MATCH (a:vacxis {entity_id: r.source_entity_id}) "
            "MATCH (b:vacxis {entity_id: r.target_entity_id}) "
            "CREATE (a)-[rel:DIRECTED]->(b) SET rel = r.properties",
            batch=rels[i:i+500],
        )

    nc = s.run("MATCH (n:vacxis) RETURN count(n) AS c").single()["c"]
    rc = s.run("MATCH (:vacxis)-[r:DIRECTED]->(:vacxis) RETURN count(r) AS c").single()["c"]
    print(f"[OK] Restored Neo4j ({db}): nodes={nc}, rels={rc}")
driver.close()
'
```

### Verify Neo4j:
```bash
uv run python -c '
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(".env")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j")),
)
with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
    nc = s.run("MATCH (n:vacxis) RETURN count(n) AS c").single()["c"]
    rc = s.run("MATCH (:vacxis)-[r:DIRECTED]->(:vacxis) RETURN count(r) AS c").single()["c"]
    print(f"nodes={nc}, rels={rc}")
driver.close()
'
```

The checked-in Neo4j export contains 3,572 nodes and 1,830 relationships. Verify the restored service separately.
