

prediction_dict = [1, 3, 5, 99, 7, 11]

futures_dict = [1.6, 3.2, 5.1, 98.9, 10.5]

with open("mot_1.txt", 'a') as f:
    f.write(str(prediction_dict))
    f.write('\n')
    f.write('`````````\n')
    f.write(str(futures_dict))
    f.write('\n')
    f.write('`````````\n')