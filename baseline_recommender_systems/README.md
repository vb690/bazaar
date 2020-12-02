## Baseline Recommender Systems


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Small project implementing baseline methodologies for recommender systems.

## Motivation

This repository arises from the need of having very basic methodologies for recommender systems. These are usefull for:

1. Benchmarking purposes.
2. Providing easyly maintainable solutions.
3. Providing fall-back solutions when more complicated methodologies fails.

## Features

**Softmax Recommender**  
Recommender engine employing a temperated softmax strategy. Recommendations are provided according to probabilities derived from observed frequencies.  
  
**Random Recommender**  
TO DO  
  
**Rule-based Recommender**  
TO DO  

## How to use  
  

**Softmax Recommender**  
We generate some synthetic data.
```python
import numpy as np

from recommendation_engines import SoftmaxRecommender


areas_mapper = {
    0: 'Action',
    1: 'Comedy',
    2: 'Horror',
    3: 'Drama',
    4: 'Thriller'
}

products_mapper = {
    'Action': {
        0: 'Indiana Jones',
        1: 'Terminator',
        2: 'The Matrix'
    },

    'Comedy': {
        0: 'Scary Movie',
        1: 'Borat',
        2: 'Hot Fuzz'
    },

    'Horror': {
        0: 'The Evil Dead',
        1: 'The Thing',
        2: 'They Live'
    },

    'Drama': {
        0: 'Bicycle Thieves',
        1: 'Patch Adams',
        2: 'Awakenings'
    },

    'Thriller': {
        0: 'Zodiac',
        1: 'Silence of the Lamb',
        2: 'Shutter Island'
    }

}

areas_dict = {
    customer_id: np.random.randint(0, 100, size=5) for
    customer_id in range(100000)
}
products_dict = {
    area_id: np.random.randint(0, 100, size=(3)) for area_id in range(5)
}
```  
We then instantiate the SoftmaxRecommender class with the generated data.
```python
recommender = SoftmaxRecommender(
    areas_dict=areas_dict,
    products_dict=products_dict,
    areas_mapper=areas_mapper,
    products_mapper=products_mapper
)

```  
And finally perform recommendations for a random sample of 15 users.
```python
for user_id in np.random.randint(0, 100000, 15):

    area, product = recommender.recommend(
        query_id=user_id,
        temperature_area=40,
        temperature_product=40
    )
    print(f'User {user_id} might like a {area} movie like {product}')
```  
Obtaining the following results:
```
User 85306 might like a Drama movie like Awakenings
User 58138 might like a Drama movie like Patch Adams
User 54383 might like a Action movie like Terminator
User 27993 might like a Thriller movie like Shutter Island
User 46247 might like a Horror movie like The Evil Dead
User 4459 might like a Horror movie like The Evil Dead
User 85215 might like a Action movie like The Matrix
User 95568 might like a Action movie like Terminator
User 1656 might like a Drama movie like Awakenings
User 55138 might like a Drama movie like Bicycle Thieves
User 78636 might like a Horror movie like The Thing
User 43027 might like a Horror movie like The Evil Dead
User 66735 might like a Comedy movie like Scary Movie
User 49284 might like a Thriller movie like Shutter Island
User 72293 might like a Action movie like The Matrix
```
 
## License

[The MIT License](https://github.com/vb690/bazaar/blob/master/LICENSE)


