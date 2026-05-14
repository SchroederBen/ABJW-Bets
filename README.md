# ABJW Bets

## Overview

ABJW Bets is an NBA betting decision-support system that uses historical game data, live matchup and odds inputs, statistical modeling, and OpenAI-assisted analysis to generate spread recommendations.

The system is designed to:

* analyze NBA matchups using historical and current data
* compare projected matchup edges against sportsbook lines
* generate a final recommendation for each game
* return either a spread pick or a PASS along with confidence and reasoning

This project is intended as a decision-support and research tool. It does not guarantee profitable bets.

## What This Branch Includes

The `feature/output-experiments` branch includes:

* the main NBA prediction pipeline
* helper modules for database queries, stat building, and AI analysis
* a simple desktop UI (`app.py`) for running the pipeline and viewing results
* project documentation and supporting code for the model/data workflow

## Repository Structure

AI/                    Main prediction pipeline and helper logic
db/                    Database-related files
docs/                  Project documentation and supporting materials
src/                   Additional data/model code
app.py                 Desktop UI for running predictions
requirements.txt       Python dependencies
README.md              Setup and run instructions

## Main Run Options

You can run the project in two ways:

### 1. Run the desktop UI

From the repo root:

py app.py

This opens the desktop interface and lets you run the prediction pipeline from a window.

### 2. Run the main pipeline directly

From the repo root:

py AI\main.py

This prints the pipeline output directly in the terminal.

## Setup Instructions

### 1. Clone the repository

Clone the repo and switch to the branch you want to use.

### 2. Install dependencies

From the repo root, install the required Python packages:

py -m pip install -r requirements.txt
py -m pip install openai requests

If `openai` and `requests` are already included in your local copy of `requirements.txt`, the second command may not be necessary.

### 3. Create a `.env` file in the repo root

The project requires environment variables for the database and OpenAI API access.

Create a file named `.env` in the root of the repository and include:

OPENAI_API_KEY=your_openai_key_here
DB_PASS=your_db_password_here
DB_USER=neondb_owner
DB_NAME=neondb
DB_HOST=your_db_host_here
DATABASE_URL=postgresql://username:password@host/dbname?sslmode=require&channel_binding=require

Notes:

* `OPENAI_API_KEY` should contain a valid OpenAI API key.
* `DB_HOST`, `DB_USER`, `DB_PASS`, and `DB_NAME` should match the Neon/PostgreSQL database being used.
* `DATABASE_URL` is also required because different parts of the project read the database configuration in different ways.

### 4. Verify the database connection

Before running the full pipeline, it is helpful to test the database connection:

py AI\db.py

If the connection works, it should print a successful connection message.

## Recommended Run Order

From the repo root:

### Test database connection

py AI\db.py

### Run the main pipeline directly

py AI\main.py

### Run the desktop UI

py app.py

## What the Pipeline Does

At a high level, the system:

1. pulls historical game and stat data from the database
2. pulls current-day NBA matchup and odds information
3. builds matchup statistics and head-to-head context
4. computes model-based edge information
5. sends structured matchup information into the AI analysis layer
6. returns a final recommendation with confidence and explanation

Typical final outputs include:

* a spread recommendation for the home or away team
* or `PASS`
* confidence
* estimated edge
* short explanation
* risk flags

## Using the Desktop UI

The desktop UI in `app.py` is a simple wrapper around the same main pipeline.

It allows you to:

* run predictions from a GUI instead of the terminal
* view the final recommendations in a cleaner format
* inspect raw output if needed

To run it:

py app.py

If the UI says no readable predictions were found, use the Show Raw Output button or run:

py AI\main.py

to verify whether the underlying pipeline produced output.

## Demo Mode

If there are no live NBA games available, the project can also be run in demo mode using sample games.

Run demo mode from the terminal

py AI\main.py --demo

Run demo mode from the desktop UI

Launch the UI normally with:
py app.py

Then click:
Run Demo

Demo mode uses sample game data from:
AI/demo_games.json

## Important Notes

* The project depends on live/current-day NBA matchup data.
* If there are no NBA games on the day you run it, the pipeline may return no predictions.
* If a live data endpoint fails, the UI may also show no predictions because it depends on the output of `AI\main.py`.
* The UI itself does not replace the main pipeline; it calls the same underlying script.

## Troubleshooting

### Error: `could not translate host name "None" to address`

The `.env` file is missing database values, is in the wrong folder, or is not being read correctly.

### UI says: `Run completed, but no predictions were found`

Possible causes:

* there are no NBA games that day
* the live API returned no games
* the main script failed before printing final results
* the output format changed and the UI parser does not match it

In that case, run:

py AI\main.py

and inspect the terminal output directly.

## Contributors

* Ben Schroeder
* Jake McConkey
* Andrew Pullen
* Wes Jennings