# LumenBridge Knowledge Management System

A fictional educational technology and consulting company uses this prototype to classify support, teaching and governance documents and retrieve relevant procedures from natural-language queries.

Repository URL:

## Run the prototype

Use Python 3.10 or newer. The core project uses only the Python standard library, with no package installation, network connection or model download required. Run these commands from this folder:

```sh
python main.py
python -m unittest discover -s tests -v
python practice.py
```

On Windows, `py -3 main.py` is an alternative if the `python` command is unavailable. Open `00_knowledge_management_prototype.ipynb` in Jupyter or Colab for the same classification and search demonstration. Its local imports require the project files to be available alongside the notebook.

Custom examples:

```sh
python main.py --query "How can a teacher adapt a reading lesson?" --output-dir evidence/custom_search
python main.py --classify-text "Check the tablet charger and record a support ticket." --output-dir evidence/custom_classification
python evidence_viewer.py
```

The optional desktop viewer requires Tkinter, which is included with most desktop Python installations. It displays saved logs without rerunning them. The command-line prototype and tests do not require Tkinter. Custom examples use separate output folders to preserve the report's recorded demonstration.

## Included project files

| Location | Purpose |
| --- | --- |
| `main.py` | Load data, train the classifier, evaluate six held-out documents and run two searches |
| `src/vectorizer.py` | Count-based TF-IDF, smoothed IDF and L2 normalization |
| `src/classification.py` | Multinomial Naive Bayes with alpha=1 smoothing |
| `src/search.py` | Rank positive cosine similarities, with document ID breaking ties |
| `src/corpus.py` | Load and validate the manifest and UTF-8 document bodies |
| `data/manifest.json` | Metadata, labels and fixed train/test split for all 18 documents |
| `data/documents/` | The complete synthetic test and training collection |
| `tests/test_system.py` | Five checks of calculations, splits, vocabulary isolation, classification and retrieval |
| `practice.py` | Six new supporting exercises and an inventory of the existing coursework |
| `evidence/` | Recorded execution, test output, JSON results and desktop screenshots |
| `docs/` | Knowledge lifecycle map and conceptual graph as editable Mermaid files |
| `report/` | Revised editable report and submission PDF |

## How it works

The classifier learns its vocabulary and IDF values from 12 training documents only, four per category. Six held-out documents are transformed with that vocabulary and assigned to Device Support, Inclusive Teaching or Governance. Document titles, labels and ownership metadata are not input features. Unknown or blank input produces `Unclassified`.

The search component builds a separate TF-IDF index over all 18 documents. It transforms a natural-language query with the same vocabulary and IDF, then ranks cosine similarities. It returns up to three positive-score matches. Blank or unknown-token queries return an empty list. This is the rubric's TF-IDF retrieval baseline: it matches shared vocabulary and does not understand arbitrary synonyms.

The corpus is small, synthetic and deliberately easy. The recorded 6/6 result demonstrates the pipeline; it does not establish real-world accuracy. Approval workflows, authentication, production access control and ontology-assisted retrieval are design proposals, not implemented features.

## Coursework notebooks

The original 16 notebooks were renamed for clarity. Their source code and saved outputs were preserved. They are supporting course artifacts, separate from the working prototype above.

| Notebook | Contents and status |
| --- | --- |
| `01_sales_data_visualization.ipynb` | Sales charts; requires the original `sales_data.csv` |
| `02_tips_exploratory_data_analysis.ipynb` | Seaborn tips data exploration; downloads a sample dataset |
| `03_world_happiness_analysis.ipynb` | Happiness analysis; requires `world_happiness_combined.csv` |
| `04_tensorflow_linear_regression.ipynb` | TensorFlow linear regression exercise |
| `05_mean_squared_error_calculation.ipynb` | Hand-calculated model error |
| `06_neural_network_experiments.ipynb` | Dense and recurrent networks, Titanic and synthetic images |
| `07_titanic_survival_classification.ipynb` | Titanic classification; requires `titanic.csv` |
| `08_light_dark_image_classification_experiments.ipynb` | Image classifier drafts; contains saved syntax errors and requires the original image dataset |
| `09_light_dark_image_classification.ipynb` | Image architecture comparison; requires the original light/dark images |
| `10_iris_clustering_and_label_propagation.ipynb` | Clustering and label propagation; requires `Iris-1.csv` |
| `11_nlp_preprocessing_and_sentiment_analysis.ipynb` | NLP and sentiment classification; contains an earlier NLTK resource error; some cells require `IMDB_dataset-1.csv` |
| `12_word_vectors_and_sentence_similarity.ipynb` | Concept coordinates and sentence embedding cosine similarity |
| `13_rule_based_chatbot.ipynb` | Pattern-based natural-language interaction |
| `14_transformer_chatbots_and_vector_routing.ipynb` | Transformer chat experiments; contains interrupted runs and downloads pretrained models |
| `15_empty_notebook.ipynb` | Empty original file; not evidence of completed work |
| `16_nlp_environment_setup.ipynb` | Installation commands only; includes a saved syntax error |

Historical notebooks use optional packages such as pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, TensorFlow, NLTK, OpenCV, PyTorch and sentence-transformers. Original external datasets were not supplied and are not fabricated here. Those notebooks are not required to reproduce the core prototype. Official Topics 1 through 6 titles were unavailable, so the report explains concrete contributions without claiming an official syllabus mapping or historical completion dates.

## Submission

Upload this project's files to your GitHub repository and enter its URL in the report's blank repository field before submission. `dist/Knowledge_Management_System_Submission.zip` is the upload package; extract it before uploading. The PDF is `report/Knowledge_Management_System_Report_Completed.pdf`.
