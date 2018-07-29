# Category classification for apps on the Apps Market

## Task
Classify apps to specific categories according to their descriptions. There are 20,104 apps with descriptions. Tf-idf value are extracted and pre-processed.

## Restriction
External libraries for classification are not allowed.

## Result
Implements Naive Bayes through python3 and reaches the average accuracy rate of 52.25%(tenfold cross validation).

## Performance
| Stage    | Time          | 
| -------- |:-------------:|
| Training | 256s          |
| Judgement| 702s          |

## Hardware
| Item      | Specification        | 
| --------  |:--------------------:|
| Processor | 2.7GHz Intel Core i5 |
| Memory    | 8GB 1867 MHz DDR3    |

## Related Assignment
USYD 2017S1 COMP5318 Asignment 1

## Code Usage Instructions

1. restore training dataset into the input folder
2. execute main.py under the algorithm folder
with python3 interpreter
3. check output folder, the predicted labels.csv
will be created after the program is nished
4. the code of ten-fold cross validation is saved
in experiment.py. if this python script is executed,
an output of average confusion matrix
will be printed in console.
5. this program is written in pycharm community
version but same IDE is not required to execute
the submitted version of program

[Report](/report.pdf)
