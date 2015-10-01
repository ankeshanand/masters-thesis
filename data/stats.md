
# Datasets

### 1. Amazon Reviews Dataset

We are using the Amazon product data available at http://jmcauley.ucsd.edu/data/amazon/ with appropriate permissions from the author. The complete dataset contains product reviews and metadata from Amazon, including 143.7 million reviews spanning May 1996 - July 2014. It  includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). 

##### Sample Review:
```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

**The dataset also has reviews aggregated by different categories. For the rest of this notebook, we will consider only the product category: Cell Phones and Accessories.**


### 2. Yelp Dataset
We use the Yelp dataset available from http://www.yelp.com/dataset_challenge/. The complete dataset contains:
* 1.6M reviews and 500K tips by 366K users for 61K businesses.
* 481K business attributes, e.g., hours, parking availability, ambience.
* Social network of 366K users for a total of 2.9M social edges.
* Aggregated check-ins over time for each of the 61K businesses.

#### Sample Review:
```
{
    'type': 'review',
    'business_id': (the identifier of the reviewed business),
    'user_id': (the identifier of the authoring user),
    'stars': (star rating, integer 1-5),
    'text': (review text),
    'date': (date, formatted like '2011-04-19'),
    'votes': {
        'useful': (count of useful votes),
        'funny': (count of funny votes),
        'cool': (count of cool votes)
    }
}
```

**Note that Yelp only provides the count of useful / funny / cool votes, and not a helpfulness score such as Amazon (2/3 users found this useful.)**

## Statistics for the Amazon Dataset


**Category:** Cell Phone and Accessories

**Total Number of Reviews from 2000 to 2013:** 2396961

**Reviews with atleast 1 helpful vote:** 721385 (30%)

#### Distribution of Helpfulness Votes
-------
**Percentage of Votes:**
![Percentage](https://raw.githubusercontent.com/ankeshanand/masters-thesis/master/data/helpfulness-votes.png "Percentage Distribution")

**Distribution among Helpful Votes:**
![Helpful](https://raw.githubusercontent.com/ankeshanand/masters-thesis/master/data/votes-distribution-without-0.png "Percentage Distribution")
