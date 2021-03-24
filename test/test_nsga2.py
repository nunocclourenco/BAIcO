import optimizers
import numpy as np 

def test_tournament():
    random_numbers = [0, 1, 2, 3, 0, 2, 1, 3]
    optimizers.np.random.randint = lambda n : random_numbers.pop(0)

    rank = [1,1,2,2]
    cdist = [0.1, 0.2, 0.1, 0.2]

    
    assert np.testing.assert_array_equal(
        optimizers.tournament(rank, cdist),
        np.array([1, 3, 0, 1])
    )== None, "should be [1,3, 0, 1]"


def test_objective_dominance():
    pop_obj = np.array([[1, 2], [2, 3],[3,2],[2,2] ])
    errors = []

    # replace assertions by conditions
    if not optimizers.objective_dominance(pop_obj, 0, 0) == (None, None):
        errors.append("Failed self non-dominance")
    
    if not optimizers.objective_dominance(pop_obj, 0, 2) == (0, 2):
        errors.append("Failed i dominance")
    
    if not optimizers.objective_dominance(pop_obj, 3, 0) == (0, 3):
        errors.append("Failed j dominance")
    
    if not optimizers.objective_dominance(pop_obj, 1, 2) == (None, None):
        errors.append("Failed non-dominance")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_assign_rank():
    dominate = [[1,2,3,4,5], [0],[0],[1,2,4,5], [1,2], [1,2]]
    dominated_by_counter = [0, 4, 4, 1, 2, 2]
   
    rank, _ = optimizers.assign_rank(dominate, dominated_by_counter)

    assert np.testing.assert_array_equal(
        rank,
        np.array([1, np.inf, np.inf, 2, 3, 3])
    )== None, "should be [1,np.inf, np.inf, 2, 3,3]"



def test_fnd_sort():
    pop_obj = np.array([[1, 2], [2, 3],[3,2],[2,2], [1,0], [0,1]  ])
    pop_cstr = np.array([0, -1, -1, 0, -0.1, -0.1])
   
    rank, _ = optimizers.fnd_sort(pop_obj, pop_cstr)

    assert np.testing.assert_array_equal(
        rank,
        np.array([1, np.inf, np.inf, 2, 3, 3])
    )== None, "should be [1,np.inf, np.inf, 2, 3,3]"



def test_crowding_distance():
    pop_obj = np.array([[1, 2], [2, 4],[0,1],[1,0], [0.2,0.8], [0.8,0.6]  ])
    rank = np.array([1, np.inf, 2, 2, 2, 2])
  

    assert np.testing.assert_allclose(
       optimizers.crowding_distance(pop_obj, rank),
       np.array([np.inf, 0, np.inf, np.inf, 1.2, 1.6])
    )== None, "should be [np.inf,0, 0, 2, np.inf,np.inf]"



def test_environment_selection():
    pop_obj = np.array([[1, 2], [2, 3],[3,2],[2,2], [1,0], [4,4], [5,3], [0,1]  ])
    pop_cstr = np.array([0, -1, -1, 0, -0.1, 0, 0, 0])
    pop = np.array([[0,0],[1,0], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0]])
    pop_meas = np.array([[10,0],[11,0], [12,0], [13,0], [14,0], [15,0], [16,0], [17,0]])


    new_pop, new_pop_obj,new_pop_cstr, new_pop_meas, front_no, crowd_dis,index = optimizers.environment_selection(pop, pop_obj, pop_cstr,pop_meas, 4)

    errors = []
    # replace assertions by conditions, anyway numpy.testing asserts will throw AssertionError
 
    if np.testing.assert_allclose(pop[index, :], new_pop) != None:
        errors.append("Failed pop")
    
    if np.testing.assert_allclose(pop_obj[index, :], new_pop_obj) != None:
        errors.append("Failed pop_obj")

    if np.testing.assert_allclose(pop_cstr[index], new_pop_cstr) != None:
        errors.append("Failed pop_cstr")
    
    if np.testing.assert_allclose(pop_meas[index,:], new_pop_meas) != None:
        errors.append("Failed pop_meas")

    if np.testing.assert_allclose(front_no, np.array([2,3,1,4])) != None:
        errors.append("Failed front_no")

    if np.testing.assert_allclose(crowd_dis, np.array([np.inf,np.inf,np.inf,np.inf])) != None:
        errors.append("Failed crowd_dis")

    if np.testing.assert_allclose(np.array(index), np.array([0,3,7,5])) != None:
        errors.append("Failed index")

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))
