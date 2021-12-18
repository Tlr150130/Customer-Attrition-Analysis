# Customer-Attrition-Analysis
Profits of a company is usually divided into 2 facets: customer acquitision and customer retention. We will focus on optimizing our customer retention effort by creating a customer attrition classifier to determine the likelihood of customer turnover. Reduction of wasted resources and increase customer retention is achieved by focusing our customer retention strategies on clients with high attrition risk.

Customer attrition was modeled using XGBoosted classifier. The model yielded a similar accuracy to the baseline model (80%), but increased recall by 80%. Automation and easy user implementation was prioritized when creating functions. GUI prototyping was implemented in the final notebook to allow for user imput.

# Improvements
### Feature Selection:
1. Feature selection was not implemented because the XGBoosted classifier possesses inate feature selection. However, better feature selection may increase the likelihood of a better model for the XGBoosted classifier

### Visualizations during EDA
1. There are countless combinations for feature interactions which yields many plots. To better organize the plots, GUI widgets could be implemented to create a tidier environment for easier understanding

### Dashboarding
1. To better facilitate deployment, a dashboard could be created to visualize the results of the predictions and updated to illistrate an overview of the client base.

