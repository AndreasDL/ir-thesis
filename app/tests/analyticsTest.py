import reporters
from main.mainAnalytics import valenceCorrelationWorker

reporter = reporters.HTMLAnalyticsReporter()
reporter.genReport( [valenceCorrelationWorker(1)] )
