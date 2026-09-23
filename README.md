
## Project Structure
- data_pipeline, analytics, and support_assistant are included in this repository.

## Project Setup

## Dependencies
- The project uses a module-level requirements file.
- support_assistant/requirements.txt contains the dependencies required for the Support Assistant module.
- The data_pipeline and analytics modules use the dependencies already available in the Python environment.

## How to Run — Data Pipeline
1. Open the data_pipeline folder.
2. Run the Python scripts in sequence according to the task documentation.
3. The generated CSV/database outputs are created by the pipeline scripts.

## How to Run — Analytics
1. Open the nalytics folder.
2. Run the Python analysis scripts.
3. The scripts generate the required analysis results and supporting artifacts.

## How to Run — Support Assistant
1. Open the support_assistant folder.
2. Install the dependencies using pip install -r requirements.txt.
3. Run python main.py to start the Support Assistant.

## Design Decisions
- **Data Pipeline:** Implemented the required data ingestion, transformation, and output steps using Python and related data-processing tools.
- **Analytics:** Used Python-based analysis and machine-learning workflows to evaluate the project data and generate supporting results.
- **Support Assistant:** Used document ingestion, retrieval, and model-based processing to provide a support-assistant workflow.
