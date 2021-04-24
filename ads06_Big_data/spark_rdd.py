
import re
from datetime import datetime as dt


# helper functions
def get_words(line):
    return re.compile(r'\w+').findall(line)


def same_att(first_att, line):
    # Get the keys of the passed in line
    att = set(line.keys())
    # check the the attributes of the first row is a subset of
    # the attributes of the passed in line
    return first_att == att


def extract_time(timestamp):
    return dt.utcfromtimestamp(timestamp)


def get_bucket(rec, min_timestamp, max_timestamp):
    interval = (max_timestamp - min_timestamp + 1) / 200.0
    return int((rec['created_at_i'] - min_timestamp)/interval)


def get_hour(rec):
    tdate = dt.utcfromtimestamp(rec['created_at_i'])
    return tdate.hour


def count_elements_in_dataset(dataset):
    """
    Given a dataset loaded on Spark, return the
    number of elements.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: number of elements in the RDD
    """
    return dataset.count()


def get_first_element(dataset):
    """
    Given a dataset loaded on Spark, return the
    first element
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: the first element of the RDD
    """
    return dataset.first()


def get_all_attributes(dataset):
    """
    Each element is a dictionary of attributes and their values for a post.
    Can you find the set of all attributes used throughout the RDD?
    The function dictionary.keys() gives you the list of attributes of a
    dictionary.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: all unique attributes collected in a list
    """
    return dataset.flatMap(lambda x: x.keys()).distinct().collect()


def get_elements_w_same_attributes(dataset):
    """
    We see that there are more attributes than just the one used in the first
    element.
    This function should return all elements that have the same attributes
    as the first element.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD containing only elements with same attributes as the
    first element
    """

    # Get the keys from the first attribute
    first_att = set(dataset.first().keys())
    return dataset.filter(lambda line: same_att(first_att, line))


def get_min_max_timestamps(dataset):
    """
    Find the minimum and maximum timestamp in the dataset
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: min and max timestamp in a tuple object
    :rtype: tuple
    """
    min_create = dataset.map(lambda line: (line['created_at_i'])).reduce(
        lambda x, y: x if x < y else y)
    min_time = extract_time(min_create)

    max_create = dataset.map(lambda line: (line['created_at_i'])).reduce(
        lambda x, y: x if x > y else y)
    max_time = extract_time(max_create)

    return (min_time, max_time)


def get_number_of_posts_per_bucket(dataset, min_time, max_time):
    """
    Using the `get_bucket` function defined in the notebook
    (redefine it in this file), this function should return a
    new RDD that contains the number of elements that fall within each bucket.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :param min_time: Minimum time to consider for buckets (datetime format)
    :param max_time: Maximum time to consider for buckets (datetime format)
    :return: an RDD with number of elements per bucket
    """

    buckets_rdd = dataset.map(lambda rec: (get_bucket(rec, min_time.timestamp(),
                                                      max_time.timestamp()), 1)).\
        reduceByKey(lambda c1, c2: c1 + c2)
    return buckets_rdd


def get_number_of_posts_per_hour(dataset):
    """
    Using the `get_hour` function defined in the notebook
    (redefine it in this file), this function should return a
    new RDD that contains the number of elements per hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with number of elements per hour
    """
    hours_buckets_rdd = dataset.map(lambda rec: (
        get_hour(rec), 1)).reduceByKey(lambda c1, c2: c1 + c2)
    return hours_buckets_rdd


def get_score_per_hour(dataset):
    """
    The number of points scored by a post is under the attribute `points`.
    Use it to compute the average score received by submissions for each hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with average score per hour
    """
    rdd1 = dataset.map(lambda rec: (get_hour(rec), (rec['points'], 1)))
    rdd2 = rdd1.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    scores_per_hour_rdd = rdd2.map(lambda hour_scores: (
        hour_scores[0], hour_scores[1][0]/hour_scores[1][1]))
    return scores_per_hour_rdd


def get_proportion_of_scores(dataset):
    """
    It may be more useful to look at sucessful posts that get over 200 points.
    Find the proportion of posts that get above 200 points per hour.
    This will be the number of posts with points > 200 divided by the total
     number of posts at this hour.
    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of scores over 200 per hour
    """
    rdd1 = dataset.map(lambda rec: (
        get_hour(rec), (1 if rec['points'] > 200 else 0, 1)))
    rdd2 = rdd1.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    prop_per_hour_rdd = rdd2.map(lambda hour_scores: (
        hour_scores[0], hour_scores[1][0]/hour_scores[1][1]))
    return prop_per_hour_rdd


def get_proportion_of_success(dataset):
    """
    Using the `get_words` function defined in the notebook to count the
    number of words in the title of each post, look at the proportion
    of successful posts for each title length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the proportion of successful post per title length
    """
    rdd1 = dataset.map(lambda rec: (len(get_words(rec['title']))
                                    if 'title' in rec.keys() else 0,
                                    (1 if rec['points'] > 200 else 0, 1)))

    rdd2 = rdd1.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

    prop_per_title_length_rdd = rdd2.map(lambda title_scores: (
        title_scores[0], title_scores[1][0]/title_scores[1][1]))
    return prop_per_title_length_rdd


def get_title_length_distribution(dataset):
    """
    Count for each title length the number of submissions with that length.

    Note: If an entry in the dataset does not have a title, it should
    be counted as a length of 0.

    :param dataset: dataset loaded in Spark context
    :type dataset: a Spark RDD
    :return: an RDD with the number of submissions per title length
    """
    rdd1 = dataset.map(lambda rec: (
        len(get_words(rec['title'])) if 'title' in rec.keys() else 0, 1))
    submissions_per_length_rdd = rdd1.reduceByKey(lambda a, b: a + b)
    return submissions_per_length_rdd
