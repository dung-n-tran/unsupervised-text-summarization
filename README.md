
# unsupervised-text-summarization
Summarize Amazon product reviews using non-convex entropy minimization. The basic idea is to select a few representative reviews that fully characterize the product features. This can be done by looking for a sparse representation of the entire dataset in an appropriate dictionary under some carefully designed criteria. This sparse representation then corresponds to the selected representatives. Nonconvex entropy minimization is used to find such representation efficiently.

Walkthrough code: [rem_amazon_review_summarization.ipynb](https://github.com/dung-n-tran/unsupervised-text-summarization/blob/master/rem_amazon_review_summarization.ipynb) (Python + Pandas)


#### Rating histogram of an Amazon product
<img src="review_rating_hist.jpg" alt="Review rating histogram" width="800"/>


#### Word-counting based summarization techniques lack of physical context
<img src="review_rating_wordcloud.jpg" alt="Review rating wordcloud" width="800"/>


#### The representative reviews (below) chosen by the proposed algorithm not only represent key features of the product but also offer physical context, e.g., written by real users, which is lacked in word-counting based techniques such as PCA.
<img src="representative_Amazon_review.jpg" alt="Representative reviews" width="800"/>

### References
DN Tran, SP Chin, TD Tran, [*"A provable nonconvex model for factoring nonnegative matrices"*](https://ieeexplore.ieee.org/document/7952559), ICASSP (2017)]
