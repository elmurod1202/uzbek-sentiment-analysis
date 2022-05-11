# uzbek-sentiment-analysis
Sentiment analysis in Uzbek language and new Datasets of Uzbek App reviews for Sentiment Classification

Feel free to use the dataset and the tools presented in this project, a paper about more details on creation and usage[here](http://www.grupolys.org/biblioteca/KurMatAloGom2019a.pdf).

If you find it useful, plese make sure to cite the paper:
```
@inproceedings{kuriyozov2019deep,
  author =       {Kuriyozov, Elmurod and Matlatipov, Sanatbek and Alonso, Miguel A and Gómez-Rodríguez, Carlos},
  title =        {Deep Learning vs. Classic Models on a New {U}zbek Sentiment Analysis Dataset},
  booktitle =     {Human Language Technologies as a Challenge for Computer Science and Linguistics – 2019},
  publisher = {Wydawnictwo Nauka i Innowacje},
  year =         {2019},
  pages = {258--262}
}
```


The main contributions of this project are:

1.  The creation of the first annotated dataset for sentiment analysis in Uzbek language, obtained from reviews of the top 100 Google Play Store applications used in Uzbekistan.  This manually annotated dataset contains  2500  positive  and  1800  negative  reviews. Furthermore, we have also built a larger dataset by automatically translating (using Google Translate API) an  existing  English  dataset of  application  reviews. The translated dataset has≈10K positive and≈10K negative app reviews, after manually eliminating themajor machine translation errors by either correctingor removing them completely.

2.  The  definition  of  the  baselines  for  sentiment  analyses in Uzbek by considering both traditional machine learning methods as well as recent deep learning techniques  fed  with  fastText  pre-trained  word  embeddings. Although all the tested models are relatively accurate  and  differences  between  models  are  small, the neural network models tested do not manage tosubstantially outperform traditional models.  We believe that the quality of currently available pre-trained word embeddings for Uzbek is not enough to let deep learning models perform at their full potential.


The results obtained through the research:
![Main Results Table](https://github.com/elmurod1202/uzbek-sentiment-analysis/blob/master/images/results-table.png)
Table:  Accuracy results with different training and test sets.ManualTT- Manually annotated Training and Test sets.TransTT- Translated Training and Test sets.TTMT- Translated dataset for Training, Annotated dataset for Test set.
