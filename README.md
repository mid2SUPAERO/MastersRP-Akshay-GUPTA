# MastersRP-Akshay-GUPTA
Scaled Aircraft in Aeroelastic Similarity - Flutter Optimization

This folder contains the following folders:

1. Garteur Model: All the python codes for all the scale ratios are inside it
2. Goland Wing: All the python codes for all the scale ratios are inside it
3. Presentation: The final defense presentation is inside this folder
4. Report: The final report in both .docx and .pdf format are inside this folder
5. Results: It contains a complied version of all the results in an excel file named "RESULT BOOK.xlsx" and each folder inside this folder contains the result of convergences and MAC matrices for respective scaling ratios

HOW TO RUN THE CODES:

To run the python codes for each scaling ratio:
1. Copy all the files from the folder "Reference Model" to the respective folder for example: Scale 1by4
2. Open the file "modal_optim_GOLAND_COBYLA_scale1by4.py" in Spyder2
3. Use python console to run the code
4. After convergence is acheived, open the file "optim_plot_COBYLA.py" in another tab on the same python window
5. Run this code in python console to get the convergence graphs
6. Go to the folder directory to read the output file "nastran_dynamic.out" for getting the frequency and mass values for scaled model
7. Go to the folder directory to read the output file "nastran_dynamic.inp" to get the optimized thickness and concentrated mass values for the scaled model
8. To get the MAC matrix type the following command in the python console: *make sure you are in the right directory*
	
	import sqlitedict
	
	db=sqlitdict.SqliteDict('modal_optim_COBYLA','iterations')
	
	db.keys()
	
	db[rank0:COBYLA|the last iteration number]['Unknowns']['MAC']
	
	
9. That's all!
