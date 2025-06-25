# What drives the price of a car?

**OVERVIEW**

In this report, I am acting a consultant to a used car dealership. The client has provided a dataset ([vehicles.csv](data/vehicles.csv)) that contains information on over 426k cars, including various details such as make, model, year, condition, and so on. My taks is to provide recommendations to my client as to what consumers value in a used car. This is a pretty vague instruction, but I will try my best!

The code and computation for this report is in the linked [Jupyter notebook](prompt_II.ipynb).

In approaching this problem, I am to follow the industry standard **CRISP-DM** framework for data science tasks:

<center>
    <img src = images/crisp.png width = 50%/>
</center>

First, we should start with a basic business understanding of what the business is trying to accomplish through this analysis.

### Business Understanding

As a used car dealership, it is important to us to understand what an appropriate market price is for the cars we sell. This dataset contains a large number of used cars with various details about them and how much they sold for. Using this large dataset, we can construct a theoretical model of how much each of these different details contributes to the market value of a car. This model can then, hopefully, be used to predict with decent precision the expected value of other cars that are not in the dataset and have not been sold yet. In doing so, the prices we set can be as attuned as possible to the details of the market, maximizing business profits.

### Data Understanding

T