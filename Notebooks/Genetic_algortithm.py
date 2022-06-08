import random
import sys
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score



class GeneSelection(object):
    # initialize variables and lists
    def __init__(self):

        self.data = pd.DataFrame
        self.genes = []
        self.parents = []
        self.newparents = []
        self.fitness_population = []
        self.best_p = []
        self.iterated = 1
        self.population = 0
        self.maxgenration = 10

        # increase max recursion for long stack
        iMaxStackSize = 15000
        sys.setrecursionlimit(iMaxStackSize)

    # create the initial population
    def initialize(self):

        for j in range(self.population):
            parent = {}
            for i in range(0, len(self.genes) - 1):
                k = random.randint(0, 1)
                parent[self.genes[i]] = k
            self.parents.append(parent)

    # set the details of this problem
    def properties(self, data, genes, population, maxgenration=10):
        self.data = data
        self.genes = genes
        self.population = population
        self.maxgenration = maxgenration
        self.initialize()

    # preparation For DataFrame
    def mapping(self, ListOfGenes, DataSet):

        data_frame = pd.DataFrame()
        for index, i in enumerate(ListOfGenes.keys()):

            if ListOfGenes[i] == 1:
                data_frame[i] = DataSet[i]

        data_frame['Class'] = DataSet['Class']
        return data_frame

    # calculate the fitness function of each list (sack)
    def fitness(self, Data):

        X = Data.drop('Class', axis=1)
        y = Data['Class']

        clf = SVC(kernel='rbf', C=1000000, gamma=0.001)  # rbf Kernel
        kf = KFold(n_splits=10, shuffle=False)
        kf.split(X)

        accuracy_model = []
        for train_index, test_index in kf.split(X):
            # Split train-test
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Train the model
            model = clf.fit(X_train, y_train)
            # Append to accuracy_model the accuracy of the model
            accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True))

        AvrgAccuracy = (sum(accuracy_model) / len(accuracy_model))
        fitness = AvrgAccuracy

        return fitness

    # run generations of GA
    def evaluatefitness(self):  # Edit

        # loop through parents and calculate fitness

        for i in range(len(self.parents)):
            parent = self.parents[i]
            newdata = self.mapping(parent, self.data)
            ft = self.fitness(newdata)
            self.fitness_population.append((ft, parent))

        return self.fitness_population

    def tournament(self, fitness_population):

        fit1, ch1 = fitness_population[random.randint(0, len(fitness_population) - 1)]
        fit2, ch2 = fitness_population[random.randint(0, len(fitness_population) - 1)]


        if (fit1 > fit2) or (fit1 == fit2 and len(ch1) < len(ch2)):
            return ch1
        else:
            return ch2

    def selectParents(self, fitness_population):

        # Construct a iterator here
        # Use Tournament Selection
        parents = []
        for i in range(0, self.population + 1, 2):
            parent1 = self.tournament(fitness_population)
            parent2 = self.tournament(fitness_population)
            parents.append(parent1)
            parents.append(parent2)

        return parents

    # mutate children after certain condition
    def mutation(self, ch):

        for j in ch.values():

            k = random.uniform(0, 1)
            if k >= 0.09:
                if j == 1:
                    j = 0
                else:
                    j = 1

        return ch

    # crossover two parents to produce two children by miixing them under random ration each time
    def crossover(self, ch1, ch2):  

        k = random.uniform(0, 1)
        if k >= 0.3:
            threshold = random.randint(1, len(ch1) - 1)
            temp1 = dict(list(ch1.items())[threshold:])
            temp2 = dict(list(ch2.items())[threshold:])
            ch1 = dict(list(ch1.items())[0:threshold])
            ch2 = dict(list(ch2.items())[0:threshold])
            ch1.update(temp2)
            ch2.update(temp1)

        else:
            pass

        return ch1, ch2

    def encodefitness(self, parent):

        newdata = self.mapping(parent, self.data)
        fit = self.fitness(newdata)
        return fit

    def NewParents(self, Parents):

        L_NewParents = []
        for Parent in Parents:
            D_NewParent = {}
            for j in Parent.keys():
                if Parent[j] == 1:
                    D_NewParent[j] = 1
                else:
                    pass
            L_NewParents.append(D_NewParent)
        return L_NewParents

    # run the GA algorithm
    def run(self):

        for i in range(0, self.maxgenration):

            print("Generation Number", i, "\n")
            # run the evaluation once
            fitness_generation = self.evaluatefitness()
            self.best_p = self.selectParents(fitness_generation)
            newparents = []
            pop = len(self.best_p) - 1

            # create a list with unique random integers
            sample = random.sample(range(pop), pop)
            for i in range(0, pop):

                # select the random index of best children to randomize the process
                if i < pop - 1:
                    r1 = self.best_p[i]
                    r2 = self.best_p[i + 1]
                    nchild1, nchild2 = self.crossover(r1, r2)
                    newparents.append(nchild1)
                    newparents.append(nchild2)
                else:
                    r1 = self.best_p[i]
                    r2 = self.best_p[0]
                    nchild1, nchild2 = self.crossover(r1, r2)
                    newparents.append(nchild1)
                    newparents.append(nchild2)

            for i in range(len(newparents)):
                newparents[i] = self.mutation(newparents[i])

            Best_fitness = 0
            Best_Parent = []
            Best_length = 1000
            for i in range(0, len(newparents)):
                fitness = self.encodefitness(newparents[i])
                ListOfActiveGenes = [j for j in newparents[i].keys() if newparents[i][j] == 1]
                if len(ListOfActiveGenes) < Best_length:
                    Best_fitness = fitness
                    Best_Parent = ListOfActiveGenes
                    Best_length = len(ListOfActiveGenes)

            print("List of Active Genes:", Best_Parent)
            print("Length of Active Genes:", len(Best_Parent))
            print("Fitness:", Best_fitness, "\n")

            self.iterated += 1
            self.parents = self.NewParents(newparents)
            self.fitness_population = []
            self.best_p = []


# LoadDataSet
Data = pd.read_csv("Modified_Series_Matrix.csv", index_col=0)

# properties for this particular problem
genes = list(Data.columns.values)
population = 100
maxgenerations = 250
GS = GeneSelection()
GS.properties(Data, genes, population, maxgenerations)
GS.run()

