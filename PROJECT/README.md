# IMDB Big Data Analysis Project

## Team Members
- **Samuel Giorno**
- **Maxime Boiral**
- **Lou Bruneau**

## Project Structure

imdb-big-data-project/
├── notebooks/
│   ├── 02_analysis_complete.ipynb    # Questions 1-14 (Batch Processing)
│   └── 03_stream_processing.ipynb    # Stream Processing
├── src/
│   ├── data_loader.py                # IMDB dataset loader
│   └── stream_processor.py           # Wikipedia stream processor
└── outputs/                          # Generated outputs (metrics.db, alerts.json)

## IMDB Datasets Used

The project automatically downloads the following datasets from https://datasets.imdbws.com/:
- name.basics.tsv.gz 
- title.basics.tsv.gz 
- title.ratings.tsv.gz
- title.crew.tsv.gz
- title.akas.tsv.gz

Datasets are downloaded automatically by notebooks/02_analysis_complete.ipynb. No manual download is required!

## Setup

```bash
pip install -r requirements.txt
```

## Instructions

1. **Run Batch Analysis** (Questions 1-14)
   - Execute all cells
   - Datasets download automatically (~2GB, takes 10-20 min first time)
   - Results answer questions 1-14

2. **Run Stream Processing**
   - Execute all cells
   - Monitors 5 IMDB entities on Wikipedia
   - Outputs: metrics.db and alerts.json

## Stream Processing Details

The stream processing part monitors on Wikipedia using the Wikimedia EventStreams recentchange endpoint:
- Christopher Nolan
- The Shawshank Redemption
- Quentin Tarantino
- The Godfather
- Steven Spielberg

For each entity, the stream processor:

- Listens to edit events on the corresponding Wikipedia pages.
- Tracks simple metrics per entity:
  - edit_count : total number of edits
  - total_bytes_changed : cumulative bytes added/removed
  - unique_users : number of distinct users
  - anonymous_edits : number of anonymous edits
  - bot_edits : number of bot edits
  - last_edit_time : timestamp of the last edit

These metrics are regularly persisted into a SQLite database:

- outputs/metrics.db  
  - **Table entity_metrics**: aggregated metrics per entity
  - **Table edit_events**: one row per edit event (entity, user, timestamp, bytes changed, comment)

### Alerting Logic

An additional **alert mechanism** is implemented:

- An alert is generated when:
  - The number of edits for an entity exceeds a threshold in the last hour (high activity).
  - An anonymous edit occurs on a tracked entity.

All alerts are written to a **separate file**:

- outputs/alerts.json (JSON Lines format)
- Each line contains: timestamp, entity, reason, and extra data about the event.

This mimics a basic alerting system, as requested in the assignment.

