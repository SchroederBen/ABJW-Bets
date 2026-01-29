# ABJW Bets — DB Setup (Docker + DBeaver)

## Why we’re doing it this way (quick)
We’re using **Docker** to run PostgreSQL so everyone has the **same database version and configuration** with one command. We’re using **DBeaver** as the GUI client so you can easily run SQL, browse tables, and debug.

**Important:** We do **not** store the actual database on GitHub. GitHub is for **code + SQL scripts** (schema/migrations). A running database contains binary data that doesn’t merge cleanly and causes conflicts/bloat. Instead, we version-control the **schema** (`db/schema.sql`) so everyone can recreate the DB consistently.

---

## Repo layout (expected)

```
ABJW-BETS/
  docker-compose.yml
  db/
    schema.sql
  src/
  docs/
  README.md
```

---

## 0) Clone the repo

```bash
git clone <OUR_REPO_URL>
cd ABJW-BETS
```

---

## 1) Install Docker Desktop (Windows)
**Why:** Docker guarantees the same Postgres setup for everyone (no “works on my machine” issues).

1. Install **Docker Desktop**
2. Open Docker Desktop and make sure it says **Running**
3. Quick check in PowerShell:

```powershell
docker --version
docker compose version
```

---

## 2) Start PostgreSQL with Docker
**Why:** This starts the Postgres server locally on your machine, identical across the team.

From the repo root (same folder as `docker-compose.yml`):

```powershell
docker compose up -d
```

Confirm it’s running:

```powershell
docker ps
```

To stop later:

```powershell
docker compose down
```

---

## 3) Install DBeaver Community
**Why:** DBeaver is an easy GUI for running SQL and viewing tables/data.

1. Download/install **DBeaver Community**
2. Open DBeaver

---

## 4) Connect DBeaver to the Docker Postgres DB
**Why:** Docker runs the database; DBeaver is the client/editor that connects to it.

In DBeaver:
1. Click **New Database Connection** (plug icon)
2. Select **PostgreSQL**
3. If prompted to download the driver, click **Download**
4. Enter the connection details below
5. Click **Test Connection** → should succeed → **Finish**

### Connection Settings

- **Host:** `localhost`
- **Port:** `5433`
- **Database:** `abjwdb`
- **Username:** `abjw`
- **Password:** `abjwpass`

> Note: We use port **5433** to avoid conflicts with any local Postgres already using 5432.

---

## 5) Initialize the schema (create tables)
**Why:** The schema is stored as SQL in the repo so everyone’s DB structure stays identical.

Schema file:
- `db/schema.sql`

### Option A (recommended): run schema from PowerShell
From repo root:

```powershell
docker exec -i abjw_postgres psql -U abjw -d abjwdb < db/schema.sql
```

### Option B: run schema from DBeaver
1. Open `db/schema.sql` in DBeaver
2. Click **Execute Script** (or highlight all → run)
3. Refresh the tree: Schemas → public → Tables → right-click **Refresh**

---

## 6) Quick test (optional)
**Why:** Confirms you’re connected and queries work.

In DBeaver SQL editor:

```sql
SELECT NOW();
```

---

# Troubleshooting

## Docker daemon not running / pipe not found
**Why it happens:** Docker Desktop isn’t open/running.

- Open **Docker Desktop**, wait until it’s running
- Retry:

```powershell
docker compose up -d
```

---

## Password authentication failed
**Why it happens:** wrong port/credentials cached OR you’re accidentally hitting another local Postgres.

- Ensure DBeaver port is **5433**
- Re-type the password (don’t rely on saved password)
- If you still can’t connect and you don’t care about wiping local DB data:

```powershell
docker compose down -v
docker compose up -d
docker exec -i abjw_postgres psql -U abjw -d abjwdb < db/schema.sql
```

---

## Port already allocated
**Why it happens:** another app is already using that port.

- We avoid this by using **5433** on the host machine.

---

## Daily workflow

Start DB:

```powershell
docker compose up -d
```

Stop DB:

```powershell
docker compose down
```
