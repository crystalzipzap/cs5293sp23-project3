# CS5293sp23 – Project3

Name: Chenyi "Crystal" Zhang

`smartcity/` - contains the pdf applications \
`project3.ipynb` - template notebook to follow for Project3

## Project Description

This project works with 69 Smart City reports and create a cluster model to explore the key topics in the files. First, the application read in the argument file name and extract the city name and state name. Then, the program proceeds to read every single reports in .pdf format and create a data frame with three columns: State, City, and "Raw Text".

The application then goes to clean up the raw texts by setting up a series of pipelines. Then the dataframe is used to test three models at different numbers of clusters, once the best model is selected, it will be used to perform topic modeling and device the themes per topic. 
## How to install

Once clone the repo, run the commands below:

```shell
pipenv install
```

```shell
pipenv shell
```

```shell
python -m nltk.downloader all
```

Note that it is extremely important to run the last command to ensure the `nltk_data` to be installed, or the project will not run no matter what.

## How to run

Run the command below to run the project:

```shell
pipenv run python project3.py --document "TX Lubbock.pdf"
```
Due to the amount of computation, this program took a long time to run and I was not able to record a .gif clip. Here is a screenshot of the output: 

![output demo](/docs/p3_demo_output_1.png)

## Test

The functions below are tested:

- extract_state_city_name(file)
- get_most_common_words(text, n=10)
- remove_most_common_words(text, most_common_words)
- remove_city_state_names(top_words, city_names, state_names)
- correct_words(words, nlp)

The other three functions are skipped because two of them are basically a wrapper functions from sklearn - assume it is called correctly and used right arguments. And the other one is the extract_file function - based on the dataframe output file, I assume it works.

```shell
pipenv run python -m pytest
```

Here is the demo for the pytest.

![output demo](/docs/p3_demo_pytest.gif)

## Bugs and Assumptions

* Ensure that the `city.pdf` file has the uniform format of `"State Abbreviation" + " " + "City name" + ".pdf"`. The code will exit if the file name cannot be found in the smartcity subdirectory
* I made the assumption that mentioning of its own city or state in a report contributes little to the clusting. But later in the result I noticed lots of city and county names in the report. I think I could have cleaned the text better.
* I accidentally ran `project3.ipynb` file after finalizing some writings and fill in blanks. I noticed later that the jupyter output and markdown filled-in-blanks have changed. I went ahead fixing the optimal k table, but I did not re-write the 36 themes based on the topic. Please note that the old answer should be fairly similar to the newer output from tne notebook.
* I notice my top 36 topics contains quite a bit of locations. My assumption is that after the cleaning functions I setup, if the location still exists, they are significant: either the city or county mentioned is succeeding in building a smart city, or there are certain institutes that contributes a lot to the progression of smart city located at that the mentioned location. A good example would be University of Wisconsin-Madison. The word "Madison" showed up a lot. 
* My code cannot output the raw and clean text into the .tsv file possibly due to the size of raw and clean text are too big. It should be able to successfully generate a .csv file based on the .ipynb output. 