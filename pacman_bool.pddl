;Header and description

(define (domain pacman_bool)

;remove requirements that are not needed
(:requirements :strips :equality :typing)
(:types 
    food
)

; un-comment following line if constants are needed
;(:constants )
;(:types food)
(:predicates 
    (enemy_around)
    (enemy_at_home) 
    (at_home) 
    (at_enemy_land) 
    (food_in_backpack)  
    (food_at_playground)
)

;define actions here

(:action go_to_enemy_playground
    :parameters ()
    :precondition (and (at_home) (not (enemy_at_home))  )
    :effect (and (not (at_home )) (at_enemy_land ))
)

(:action go_to_enemy
    :parameters ()
    :precondition (and 
        (at_home)
    )
    :effect (and 
        (enemy_around)
    )
)

(:action eat_enemy
    :parameters ()
    :precondition (and (at_home ) (enemy_around))
    :effect (and 
    (not (enemy_around))
    (not (enemy_at_home))
    )
)

(:action eat_food
    :parameters ()
    :precondition (and (not (enemy_around)) (not (enemy_at_home)) (at_enemy_land ) (food_at_playground)    )
    :effect (and 
        (not (food_at_playground))
        (food_in_backpack)
    )
)
(:action go_home
    :parameters ()
    :precondition (and (at_enemy_land ) )
    :effect (and 
        (not (at_enemy_land ))
        (at_home )
    )
)

(:action unpack_food
    :parameters ()
    :precondition (and (at_home))
    :effect (and 
        (not (food_in_backpack))
    )
    
    
)


)
