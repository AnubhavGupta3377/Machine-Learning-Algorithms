# Project Title
- Implementation of Decision Tree from scratch in Python

## Input format
- All data is in SSV file format
- It is a simple text format, consisting of lines of either administrative information (the "header", first 3 lines), or data lines (the rest).  Each line consists of a number of words.  There is an arbitrary number of spaces
or tabs allowed between words.  However, reasonably, a line cannot contain newlines.

- Header (first 3 lines):
  - The first line contains two numbers, the number of fields (attributes, target attribute included) and the number of 0 (included for reasons of backwards compatability - please do not modify or remove). The second line contains as many words as are fields. Each word represents the name of the attribute. The third line contains as many characters as attributes. Each character is either 'c' (continuous attribute), 'b' (binary, 0/1 attribute) or 'd' (discrete attribute, more than two alternatives).

- Data (rest):
  - The rest of the file contains the data, with one example per line. Note that binary attributes can only be represented with the two numbers 0 and 1.  Discrete attributes can contain an arbitrary number of values, each corresponding to a different string.  The "dt" program automatically deduces the cardinality of each discrete attribute.

- NOTE: the target attribute is ALWAYS the first column and can only be binary.
- Note that this is a rigid format, and you should make sure to follow it if you decide to add additional data.

## Implementation Details
- This implementation of decision tree works only when all attributes are discrete valued or binary valued.
- Only performs **binary classifications**

## Author
- **Anubhav Gupta**

## Prerequisites
- Python 2.7
