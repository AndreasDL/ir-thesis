import reporters
from archive.mainAnalytics import valenceCorrelationWorker

reporter = reporters.HTMLAnalyticsReporter()
reporter.genReport( [valenceCorrelationWorker(1)] )
