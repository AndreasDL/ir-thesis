import reporters
from archive.mainCorrSelection import valenceCorrAccWorker

reporter = reporters.HTMLCorrReporter()
results = valenceCorrAccWorker(1)
reporter.genReport( [results] )