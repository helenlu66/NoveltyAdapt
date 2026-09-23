(define (problem kitchen)
  (:domain kitchen)
  (:objects
    lid1 - lid
    pot1 - pot
    bread1 - bread
    stove1 - stove
    switch1 - switch
    table1 - table
    gripper1 - gripper
  )

  ;initial symbolic state of the task using ONLY available predicates
  (:init
    (directly-on-table bread1 table1)
    (directly-on-table stove1 table1)
    (directly-on-table switch1 table1)
    (pickupable pot1)
    (pickupable bread1)
    (pickupable lid1)
    (free gripper1)
    (ontop lid1 pot1)
    (ontop pot1 stove1)
    (covered pot1)
  )

  (:goal 
    (and
      (in bread1 pot1)
      (on stove1)
      (ontop pot1 stove1)
    )
  )

)