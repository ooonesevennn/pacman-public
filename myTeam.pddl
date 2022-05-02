;Header and description

(define (domain pacman_bool)

    ;remove requirements that are not needed
    (:requirements :strips :typing)
    (:types 
        enemy1 - enemy
        enemy2 - enemy
        ally - team
        current_agent - team
    )

    ; un-comment following line if constants are needed
    ;(:constants )
    ;(:types food)
    (:predicates 

        ;Basic predicates
        (enemy_around ?e - enemy ?a - team) ;enemy ?e is within 4 grid distance with agent ?a
        (is_pacman ?x) ; if an agent is pacman 
        (food_in_backpack ?a - team)  ; have food in backpack
        (food_available) ; still have food on enemy land

        ;Predicates for virtual state to set goal states
        (defend_foods) ;The environment do not collect state for this predicates, this is a virtual effect state for action patrol 


        ;Advanced predicates
        ;These predicates are currently not used and consider the state of other agent 
        (enemy_long_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 25 
        (enemy_medium_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 15 
        (enemy_short_distance ?e - enemy ?a - current_agent) ; noisy distance return shorter than 15 

        (food_backpack_gt3 ?a - team) ; more than 3 food in backpack
        (food_backpack_gt5 ?a - team)  ; more than 5 food in backpack
        (food_backpack_gt10 ?a - team)    ; more than 10 food in backpack
        (food_backpack_gt20 ?a - team)    ; more than 20 food in backpack

        (near_food ?a - current_agent)  ; a food within 4 grid distance 
        (near_capsule ?a - current_agent)   ;a capsule within 4 grid distance
        (capsule_available) ; capsule available on map
        (winning)   ; is the team score more than enemy team
        (winning_gt3) ; is the team score 3 more than enemy team
        (winning_gt5)    ; is the team score 5 more than enemy team
        (winning_gt10)  ; is the team score 10 more than enemy team
        (winning_gt20)  ; is the team score 20 more than enemy team
        (near_ally  ?a - current_agent) ; is ally near 4 grid distance
        (is_scared ?x) ;is enemy, current agent, or the ally in panic (due to capsule eaten by other side)


        ;Cooperative predicates
        ;The states of the following predicates are not collected by demo team_ddl code;
        ;To use these predicates, You have to collect the corresponding states when preparing pddl states in the code.
        ;These predicates describe the current action of your ally
        (eat_enemy ?a - ally)
        (go_home ?a - ally)
        (go_enemy_land ?a - ally)
        (eat_capsule ?a - ally)
        (eat_food ?a - ally)

    )

    ;define actions here

    (:action go_to_enemy_land
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2 )
        :precondition (and (not (is_pacman ?a)) (not (is_pacman ?e1)) (not (is_pacman ?e2)) (not (enemy_around ?e1 ?a)) (not (enemy_around ?e2 ?a))  )
        :effect (and (is_pacman ?a))
    )

    (:action chase_enemy_at_home
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and 
            (not (is_pacman ?a))
            (is_pacman ?e)
        )
        :effect (and 
            (enemy_around ?e ?a)
        )
    )

    (:action eat_enemy_at_home
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and (not (is_pacman ?a)) (enemy_around ?e ?a) (is_pacman ?e))
        :effect (and 
        (not (enemy_around ?e ?a))
        (not (is_pacman ?e))
        )
    )

    (:action fallback
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and 
            (not (is_pacman ?a))
            (enemy_around ?e ?a)
            (not (is_pacman ?e))
        )
        :effect (and 
            (is_pacman ?e)
        )
    )

    (:action eat_food
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and (not (enemy_around ?e1 ?a)) (not (is_pacman ?e1)) (not (enemy_around ?e2 ?a)) (not (is_pacman ?e2)) (is_pacman ?a)  (food_available)    )
        :effect (and 
            (not (food_available))
            (food_in_backpack ?a)
        )
    )
    (:action go_home
        :parameters (?a - current_agent)
        :precondition (and (is_pacman ?a) )
        :effect (and 
            (not (is_pacman ?a))
        )
    )

    (:action unpack_food
        :parameters (?a - current_agent)
        :precondition (and (not (is_pacman ?a)))
        :effect (and 
            (not (food_in_backpack ?a))
            (not (20_food_in_backpack ?a))
            (not (10_food_in_backpack ?a))
            (not (5_food_in_backpack ?a))
            (not (3_food_in_backpack ?a))
        )
    )

    (:action patrol
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (not (is_pacman ?a))
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
            (winning_gt5)
        )
        :effect (and 
            (defend_foods)
        )
    )

)