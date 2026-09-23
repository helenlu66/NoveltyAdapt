(define (domain kitchen)
    (:requirements :strips :typing)
    (:types 
        gripper table tabletop-object - object
        lid bread pot on-off-object - tabletop-object
        switch stove - on-off-object
    ) 
    
    ;; Define predicates
    (:predicates
        (pickupable ?tabletop-object - tabletop-object) ; whether the object can be picked up.
        (directly-on-table ?tabletop-object - tabletop-object ?table - table) ; whether the object is on the table directly making contact with the table.
        (ontop ?tabletop-object - tabletop-object ?tabletop-object - tabletop-object) ; whether one object is on top of another object.
        (in ?tabletop-object - tabletop-object ?pot - pot) ; whether the object is inside the pot.
        (covered ?pot - pot) ; whether the pot is covered.
        (on ?on-off-object - on-off-object) ; whether the object is on.
        (exclusively-occupying-gripper ?tabletop-object - tabletop-object ?gripper - gripper) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (free ?gripper - gripper) ; whether the gripper is not occupied by anything. If true, there should be no true `exclusively-occupying-gripper` atoms.
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-from-tabletop
        :parameters (?obj - tabletop-object ?table - table ?gripper - gripper) 
        :precondition (and (directly-on-table ?obj ?table) (pickupable ?obj) (free ?gripper)) 
        :effect (and (exclusively-occupying-gripper ?obj ?gripper) (not (free ?gripper)) (not (directly-on-table ?obj ?table)))
    ) 

    (:action place-on-table
        :parameters (?obj - tabletop-object ?table - table ?gripper - gripper)
        :precondition (and (exclusively-occupying-gripper ?obj ?gripper) (not (directly-on-table ?obj ?table)) (pickupable ?obj) (not (free ?gripper)))
        :effect (and (directly-on-table ?obj ?table) (not (exclusively-occupying-gripper ?obj ?gripper)) (free ?gripper))
    )
    

    (:action place-in-pot
        :parameters (?bread - bread ?pot - pot ?gripper - gripper)
        :precondition (and (not (covered ?pot)) (exclusively-occupying-gripper ?bread ?gripper) (not (in ?bread ?pot)) (pickupable ?bread) (not (free ?gripper)))
        :effect (and (in ?bread ?pot) (not (exclusively-occupying-gripper ?bread ?gripper)) (free ?gripper)) 
    )

    (:action place-on-object
        :parameters (?obj1 - tabletop-object ?obj2 - tabletop-object ?gripper - gripper)
        :precondition (and (exclusively-occupying-gripper ?obj1 ?gripper) (pickupable ?obj1) (not (ontop ?obj1 ?obj2)) (not (ontop ?obj2 ?obj1)) (not (free ?gripper)) (not (exclusively-occupying-gripper ?obj2 ?gripper)))
        :effect (and (ontop ?obj1 ?obj2) (not (exclusively-occupying-gripper ?obj1 ?gripper)) (free ?gripper))
    )

    (:action turn-on-stove
        :parameters (?switch - switch ?stove - stove ?gripper - gripper)
        :precondition (and 
            (not (on ?stove))
            (not (on ?switch))
            (free ?gripper)
        )
        :effect (and 
            (on ?stove)
            (on ?switch)
            (exclusively-occupying-gripper ?switch ?gripper)
            (not (free ?gripper))
        )
    )
    (:action free-gripper-from-fixed-object
        :parameters (?obj - tabletop-object ?gripper - gripper)
        :precondition (and (exclusively-occupying-gripper ?obj ?gripper) (not (pickupable ?obj)) (not (free ?gripper)))
        :effect (and (not (exclusively-occupying-gripper ?obj ?gripper)) (free ?gripper))
    )
    
    
)