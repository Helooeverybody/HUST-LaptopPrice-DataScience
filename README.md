# HUST-LaptopPrice-DataScience

## Project Description

Laptops play a crucial role in modern life, serving a wide range
of purposes, from casual browsing to high-end computational tasks.
To provide a valuable insight of the laptop market, this project fo-
cuses on exploring the complex relationships between various lap-
top specifications, such as CPU, GPU, RAM, storage, and display,
as well as their influence on price. By analyzing a massive manually
collected dataset, we uncover how different components interact and
contribute to overall performance and pricing. The study includes de-
tailed information on the correlations among key features, trends in
price segments, and the identification of the most impactful specifica-
tions. Furthermore, multiple machine learning models are developed
to predict laptop prices and performance scores based on their speci-
fications. The findings are presented through intuitive visualizations,
enabling manufacturers and consumers to make data-driven decisions
regarding design, purchasing, and marketing strategies

## Folder structures

```
.
├── data/                                 # the data .csv file should be inside this folder
|   └── data_updating_tools               # tools used to update the data. Please refer to the UI for this part
├── pred/
|   ├── display_score.inpynb              # display score regression
|   ├── work_score.inpynb                 # work score regression
|   ├── portability_score.inpynb          # portability score regression
|   ├── play_score.inpynb                 # play score regression
|   └── cost.inpynb                       # cost regression
├── insight/
|   ├── basic_visulaization.ipynb         # basic visualization of data
|   └── eda.ipynb                         # exploratory data analysis
├── scraping&crawling/                    # scraping and crawling folder
|   ├── scraping.ipynb                    # main file used to scrape the data
├── cleaning&integration/
|   ├── clean/                            # notebooks for cleaning the data
|   └── integrate/                        # notebooks for integrating CPU&GPU into Laptop
|   └── pipeline.py                       # the pipeline file to do both cleaning and integrating
├── gui/
|   ├──                                   #
|   └──                                   #


```

## UI guide
