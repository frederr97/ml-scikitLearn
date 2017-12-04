import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

datasets = ['dataset_31_credit-g', 'dataset_53_heart-statlog', 'ionosphere', 'wdbc']
for dataset in datasets:
	print 'Dataset:', dataset
	df = pd.read_csv('Datasets//' + dataset + '.csv', sep=',')
	data = df.drop(df.columns[[-1]], axis=1)
	if dataset == 'dataset_31_credit-g':
		data = pd.get_dummies(data)
	target = df[df.columns[-1]]
	data_train, data_test, target_train, target_test = train_test_split(
		data, target, test_size=0.7, random_state=0)
	nomes_classificadores =[
		'Arvore de decisao ',
		'KNN               ',
		'Naive-Bayes       ',
		'Redes Neurais     '#,
		#'SVM               '
		]
	classifiers = [
		DecisionTreeClassifier(max_depth=5),
    	KNeighborsClassifier(3),
    	GaussianNB(),
    	MLPClassifier(alpha=1)#,
    	#SVC(kernel="linear", C=0.025),
		]
	for nome, classifier in zip(nomes_classificadores, classifiers):
		print ' ', nome
		#scores = cross_val_score(classifier, data, target, cv=3)
		#fo_star = scores.mean()
		features = []
		for i in data: features.append(1)
		classifier.fit(data_train, target_train)
		fo_star = classifier.score(data_test, target_test)
		s_star = features[:]
		s_tmp = s_star[:]
		fo_tmp = fo_star
		aux = ''
		for i in s_star: aux = aux + str(i)
		print '  | S I: %.2f%% %s' % (fo_star, aux)
		"""
		for i in range(0, len(features)):
			features[i] = 0
			#scores = cross_val_score(classifier, data_tmp, target, cv=3)
			#fo = scores.mean()
			data_train_tmp = data_train.drop(data_train.columns[[i]], axis=1)
			classifier.fit(data_train_tmp, target_train)
			data_test_tmp = data_test.drop(data_test.columns[[i]], axis=1)
			fo = classifier.score(data_test_tmp, target_test)
			if fo > fo_tmp:
				fo_tmp = fo
				s_tmp = features[:]
				aux = ''
				for j in s_tmp: aux = aux + str(j)
				print '  |  M1:', aux, fo_tmp
				#print features
			features[i] = 1
		if fo_tmp > fo_star:
			fo_star = fo_tmp
			s_star = s_tmp[:]
		features = s_star
		"""
		melhoria = True
		cont = 1
		while melhoria:
			melhoria = False
			for i in range(0, len(features)):
				if features[i] == 0:
					continue
				features[i] = 0
				data_train_tmp = data_train.drop(data_train.columns[[i]], axis=1)
				#scores = cross_val_score(classifier, data_tmp, target, cv=3)
				#fo = scores.mean()
				classifier.fit(data_train_tmp, target_train)
				data_test_tmp = data_test.drop(data_test.columns[[i]], axis=1)
				fo = classifier.score(data_test_tmp, target_test)
				if fo > fo_tmp:
					fo_tmp = fo
					s_tmp = features[:]
					aux = ''
					for i in s_tmp: aux = aux + str(i)
					print '  |  m%d:%.2f%% %s' % (cont, fo_tmp * 100, aux)
					#print features
				features[i] = 1
			if fo_tmp > fo_star:
				fo_star = fo_tmp
				s_star = s_tmp
				aux = ''
				for i in s_tmp: aux = aux + str(i)
				print '  |  M%d:%.2f%% %s' % (cont, fo_tmp * 100, aux)
				melhoria = True
				cont = cont + 1
		print '  \\%.4f%%' % (fo_star * 100)
		print nome, ("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))