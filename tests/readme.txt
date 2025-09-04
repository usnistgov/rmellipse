This folder contains unit tests

Put test data you want versioned, and to never change, inside const/

Anything in mutable/ should not be expected to be the same between calls
to test functions, and will not be versioned. Use it to store temporary outputs
during test function runs.
