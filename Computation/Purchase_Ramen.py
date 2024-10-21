def monopolize_ramen():
    """Greedy Algorithm for monopolizing ramen"""
    # 언제 3개를 사야 하는지
    # 언제 2개를 사야 하는지
    # 1 2 1 1 vs 2 4 2 2
    
    import sys
    ### Read Input
    readline = sys.stdin.readline
    num_factories = int(readline().strip())
    factories = list(map(int, readline().strip().split(" ")))
    # enumerate_factories = enumerate(factories)
    
    cumprice = 0 # cumulative price
    index = 0
    
    # for ramen_index, ramen in enumerate_factories:
    while index < num_factories:
        kimochii = 0 # A boolean for whether ramen at current iteration is depleted
        price = 0
        skip_counter = 0
        ramen = factories[index]
        broken = False
        
        # 2 1 2-> 1 1 2
        # 1 3 1 2->1 2 1 2-> 0 1 1 2->0 0 0 1->
        # 2 4 2 2->1 3 2 2-> 0 2 2 2->
        
        while(not bool(kimochii)):
            if ramen > 0:
                if not broken:
                    price += 3
                    ramen -= 1
                    kimochii = int(ramen <= 0)
                    previous_depleted = kimochii
                    current_not_depleted = int(kimochii == 0)
                    try:
                        for i in range(index + 1, index + 3):
                            print(f"previous depleted: {previous_depleted}, index: {index}")
                            if previous_depleted == 1:
                                skip_counter += 1
                            current_not_depleted = int(factories[i] > 1)
                            multiplying_factor = int((previous_depleted == 1 or current_not_depleted == 1) and factories[i] > 0)
                            price += 2*multiplying_factor
                            if multiplying_factor == 1:
                                factories[i] -= 1
                            else:
                                broken = True
                                # print("broken!")
                                break
                            # print("fuck")
                            previous_depleted = int(current_not_depleted == 0)
                    except IndexError:
                        # in order to account for the case when current factory is towards the end index but still has some ramen
                        pass
                else:
                    break
            else:
                kimochii = 1
            
            # Update factories
            factories[index] = ramen
        
        print(f"factories: {factories}")
        print(f"added price: {price}, index: {index}")
        print(f"skip counter: {skip_counter}")
        
        # update index
        if not broken:
            index = index + max(1, skip_counter) if index < num_factories - 2  else index + max(1, skip_counter) % num_factories
        cumprice += price
        print(f"next index: {index}")
        
    print(cumprice)
    
monopolize_ramen()