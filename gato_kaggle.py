import csv

with open('submission.csv', mode='w') as file:
    writer = csv.writer(file, delimiter='')
    writer.writerow(['id', 'valor'])