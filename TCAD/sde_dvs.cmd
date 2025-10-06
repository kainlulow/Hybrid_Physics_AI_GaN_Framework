;Reinitializing SDE
(sde:clear)
;Set coordinate system up direction
(sde:set-process-up-direction "+z")

(define tpInGaN @tpInGaN@)
;(define tpInGaN 0.06)
(define tRecess @tRecess@)
;(define tRecess 0.03)
(define tAlGaN 0.015)
(define tGaN 1)
(define tsi 0.5)
(define tox 0.005)
;(define tp++InGaN 0.003)
(define tp++InGaN @tp++InGaN@)

#if @AlN1@ == 1
(define tAlN_1 0.002)
#else 
(define tAlN_1 0.000)
#endif 

#if @AlN2@ == 1
(define tAlN_2 0.001)
#else
(define tAlN_2 0.000)
#endif

(define Ls 0.45)
(define Ld 0.45)
(define Lsg 1)
(define Ldg 1)
(define Lch 2)
(define L (+ Ls Lsg Lch Ldg Ld))

(define pInGaN_doping @InGaN_Dope@) ;3e17)
(define p++InGaN_doping @p+_InGaN_Dope@) ;2e18)

(define InGaN_mole @xIn@)

(define AlGaN_doping 1e15)
(define AlGaN_mole @xAl@)

(define GaN_doping 1e15)
(define GaN_mole 0.05)


;================ stucture ===========================
;-------p++InGaN layer-------

(sdegeo:create-rectangle 
  (position 0 0 0) 
  (position L tp++InGaN 0)
  "InGaN" "p++InGaN"
)

(sdedr:define-constant-profile "p++InGaN_const" "BoronActiveConcentration" p++InGaN_doping)
(sdedr:define-constant-profile-region "p++InGaN_const" "p++InGaN_const" "p++InGaN")

(sdedr:define-constant-profile "xmole_p++InGaN" "xMoleFraction" InGaN_mole)
(sdedr:define-constant-profile-region "xmole_p++InGaN_region" "xmole_p++InGaN" "p++InGaN")


;----------p-InGaN----------
(sdegeo:create-rectangle 
  (position 0 tp++InGaN 0)
  (position L (+ tp++InGaN tpInGaN) 0)
  "InGaN" "pInGaN"
)

(sdedr:define-constant-profile "pInGaN_const" "BoronActiveConcentration" pInGaN_doping)
(sdedr:define-constant-profile-region "pInGaN_const" "pInGaN_const" "pInGaN")

(sdedr:define-constant-profile "xmole_pInGaN" "xMoleFraction" InGaN_mole)
(sdedr:define-constant-profile-region "xmole_pInGaN_region" "xmole_pInGaN" "pInGaN")

#if @AlN1@ == 1
;------------AlN_1---------------
(sdegeo:create-rectangle 
  (position 0 (+ tp++InGaN tpInGaN) 0)
  (position L (+ tp++InGaN tpInGaN tAlN_1) 0)
  "AlN" "AlN_1"
)
#endif 

;------------AlGaN-----------------
(sdegeo:create-rectangle 
  (position 0 (+ tp++InGaN tpInGaN tAlN_1) 0)
  (position L (+ tp++InGaN tpInGaN tAlN_1 tAlGaN) 0)
  "AlGaN" "AlGaNb"
)

(sdegeo:bool-unite (find-region-id "AlGaNb"))
(sdedr:define-constant-profile "ndop_AlGaN_const" "ArsenicActiveConcentration" AlGaN_doping)
(sdedr:define-constant-profile-region "ndop_AlGaN_const" "ndop_AlGaN_const" "AlGaNb")
(sdedr:define-constant-profile "xmole_AlGaN_const" "xMoleFraction" AlGaN_mole)
(sdedr:define-constant-profile-region "xmole_AlGaN_const" "xmole_AlGaN_const" "AlGaNb")

#if @AlN2@ == 1
;-------------AlN_2------------------
(sdegeo:create-rectangle 
  (position 0 (+ tp++InGaN tpInGaN tAlN_1 tAlGaN) 0)
  (position L (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2) 0)
  "AlN" "AlN_2"
)
#endif

;-------------GaN buffer-----------------
(sdegeo:create-rectangle 
  (position 0 (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2) 0) 
  (position L (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2 tGaN) 0)
  "GaN" "GaNb"
)
(sdedr:define-constant-profile "ndop_GaN_const" "ArsenicActiveConcentration" GaN_doping)
(sdedr:define-constant-profile-region "ndop_GaN_const" "ndop_GaN_const" "GaNb")
(sdedr:define-constant-profile "xmole_GaN_const" "xMoleFraction" GaN_mole)
(sdedr:define-constant-profile-region "xmole_GaN_const" "xmole_GaN_const" "GaNb")


;--------------Si substrate--------------------
(sdegeo:create-rectangle 
  (position 0 (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2 tGaN) 0)
  (position L (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2 tGaN tsi) 0)
  "Silicon" "Si_sub"
)


;----------------Oxide--------------------
(sdegeo:set-default-boolean "ABA")

(sdegeo:create-rectangle
	(position Ls (- 0 tox) 0)
	(position (+ Ls Lsg) 0 0)
	"Al2O3" "R.LeftOxide"
	)

(sdegeo:create-rectangle
	(position (+ Ls Lsg) (- 0 tox) 0)
	(position (+ Ls Lsg Lch) (+ tp++InGaN tRecess) 0)
	"Al2O3" "R.GateOxide"
	)

(sdegeo:create-rectangle
	(position (+ Ls Lsg Lch) (- 0 tox) 0)
	(position (+ Ls Lsg Lch Ldg) 0 0)
	"Al2O3" "R.RightOxide"
	)


;=============== Contact ====================
######  Source  #######
(sdegeo:create-rectangle 	
	(position 0 (- 0 0.2) 0) 
	(position Ls 0  0) 
	"Aluminum"  "Reg.source" )


(sdegeo:define-contact-set "source" 10 (color:rgb 1 0 0 ) "##" )
(sdegeo:set-current-contact-set "source")
(sdegeo:set-contact-boundary-edges (list(car (find-body-id (position 0  0  0) )) ) "source")

(sdegeo:delete-region (list(car (find-body-id (position 0 (- 0 0.05)   0) )) ) )


###### Drain #######
(sdegeo:create-rectangle 	
	(position (+ Ls Lsg Lch Ldg) (- 0 0.2) 0) 
	(position L 0  0) 
	"Aluminum"  "Reg.drain" )

(sdegeo:define-contact-set "drain" 10 (color:rgb 1 0 0 ) "##" )
(sdegeo:set-current-contact-set "drain")
(sdegeo:set-contact-boundary-edges (list(car (find-body-id (position L  0  0) )) ) "drain")
(sdegeo:delete-region (list(car (find-body-id (position L (- 0 0.05)   0) )) ) )


######  Gate  ########
(sdegeo:set-default-boolean "ABA")
(sdegeo:create-rectangle 	
	(position (+ Ls Lsg tox) (- 0 0.2) 0)
	(position (- (+ Ls Lsg Lch) tox) (- (+ tp++InGaN tRecess) tox)  0) 
	"Aluminum"  "Reg.gate" )

(sdegeo:define-contact-set "gate" 10 (color:rgb 1 0 0 ) "##" )
(sdegeo:set-current-contact-set "gate")
(sdegeo:set-contact-boundary-edges (list(car (find-body-id (position (+ Ls Lsg tox) (- 0 0.2)  0) )) ) "gate")
(sdegeo:delete-region (list(car (find-body-id (position (+ Ls Lsg tox) (- 0 0.1) 0) )) ) )




;================== Mesh ===================
(sdegeo:set-default-boolean "ABA")

(sdedr:define-refeval-window "Pl.global" "Cuboid"
                (position 0 0 0) 
                (position L (+ tp++InGaN tpInGaN tAlN_1 tAlGaN tAlN_2 tGaN tsi) 0))

(define minsg 0.05)
(define maxsg 0.2)
(sdedr:define-refinement-size "Ref.global"
	maxsg maxsg maxsg
	minsg minsg minsg)

(sdedr:define-refinement-function "Ref.global" "DopingConcentration" "MaxTransDiff" 1)

(sdedr:define-refinement-placement "Ref.global" "Ref.global" "Pl.global")


(define minsg 0.01)
(define maxsg 0.05)
(sdedr:define-refinement-size "Ref.Hetero"
	maxsg maxsg maxsg
	minsg minsg minsg)

(define mins 0.001)
(define maxs 0.01)

#if @AlN1@ == 1
(define mins0 0.001)
(define maxs0 0.005)
(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "InGaN" "AlN" mins0 maxs0 "DoubleSide")

;---Region: InGaN/AlN ---
(define LTmin (+ tp++InGaN (* tpInGaN 0.5)))
(define LTmax (+ tp++InGaN tpInGaN (* tAlN_1 0.5)))

(sdedr:define-refeval-window
    "Rwin.InGaN_AlN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)

(sdedr:define-refinement-placement "RPInGaN_AlN.Hetero" "Ref.Hetero" "Rwin.InGaN_AlN")


(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "AlN" "AlGaN" mins maxs "DoubleSide")
;---Region: AlN/AlGaN ---
(define LTmin (+ tp++InGaN tpInGaN (* tAlN_1 0.5)))
(define LTmax (+ tp++InGaN tpInGaN tAlN_1 (* tAlGaN 0.5)))

(sdedr:define-refeval-window
    "Rwin.AlN_AlGaN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)
(sdedr:define-refinement-placement "RPAlN_AlGa.Hetero" "Ref.Hetero" "Rwin.AlN_AlGaN")


#else
(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "InGaN" "AlGaN" mins maxs "DoubleSide")
;---Region: INGaN_AlGaN ---
(define LTmin (+ tp++InGaN (* tpInGaN 0.5)))
(define LTmax (+ tp++InGaN tpInGaN tAlN_1 (* tAlGaN 0.5)))

(sdedr:define-refeval-window
    "Rwin.lnGaN_AlGaN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)
(sdedr:define-refinement-placement "RPlnGaN_AlGaN.Hetero" "Ref.Hetero" "Rwin.lnGaN_AlGaN")

#endif


#if @AlN2@ == 1
(define mins0 0.001)
(define maxs0 0.005)
(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "AlN" "AlGaN" mins0 maxs0 "DoubleSide")

;---Region: AlGaN/AlN ---
(define LTmin (+ tp++InGaN tpInGaN tAlN_1 (* tAlGaN 0.5)))
(define LTmax (+ tp++InGaN tpInGaN tAlN_1 tAlGaN (* tAlN_2 0.5)))

(sdedr:define-refeval-window
    "Rwin.AlGaN_AlN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)

(sdedr:define-refinement-placement "RPAlGaN_AlN.Hetero" "Ref.Hetero" "Rwin.AlGaN_AlN")


(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "AlN" "GaN" mins0 maxs0 "DoubleSide")
;---Region: AlN_GaN ---

(define LTmin (+ tp++InGaN tpInGaN tAlN_1 tAlGaN (* tAlN_2 0.5)))
(define LTmax (+ tp++InGaN tpInGaN tAlN_1 tAlGaN  tAlN_2 (* tGaN 0.001)))
(sdedr:define-refeval-window
    "Rwin.AlN_GaN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)
(sdedr:define-refinement-placement "RPAlN_GaN.Hetero" "Ref.Hetero" "Rwin.AlN_GaN")


#else
(sdedr:define-refinement-function "Ref.Hetero" "MaxLenInt" "AlGaN" "GaN" mins maxs "DoubleSide")

;---Region: AlGaN_GaN ---
(define LTmin (+ tp++InGaN tpInGaN tAlN_1 (* tAlGaN 0.5)))
(define LTmax (+ tp++InGaN tpInGaN tAlN_1 tAlGaN  tAlN_2 (* tGaN 0.001)))

(sdedr:define-refeval-window
    "Rwin.AlGaN_GaN"
    "Rectangle"
    (position 0 LTmin 0)
    (position L LTmax 0)
)
(sdedr:define-refinement-placement "RPAlGaN_GaN.Hetero" "Ref.Hetero" "Rwin.AlGaN_GaN")
#endif


(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "pInGaN" "source" 0.0005 1.2 "UseRegionNames")
(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "pInGaN" "gate" 0.0005 1.2 "UseRegionNames")
(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "pInGaN" "drain" 0.0005 1.2 "UseRegionNames")



;(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "GaNchannel" "pInGaN" 0.001 1.8 "UseRegionNames")
;(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "pInGaN" "AlGaNb" 0.001 1.8 "DoubleSide" "UseRegionNames")
;(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "AlGaNb" "GaNb" 0.001 1.8 "DoubleSide" "UseRegionNames")
(sdedr:define-refinement-function "Ref.global" "MaxLenInt" "InGaN" "Al2O3" 0.001 1.8 "DoubleSide")

;(sdedr:define-refeval-window "Pl.channel" "Cuboid"
;                (position 2.1 0 0) (position 4.5 0.2 0.0))
;(sdedr:define-refinement-size "Ref.channel" 0.5 0.01  0.05 0.01 0.001  0.001)

;(sdedr:define-refinement-function "Ref.channel" "DopingConcentration" "MaxTransDiff" 1)
;(sdedr:define-refinement-function "Ref.channel" "MaxLenInt" "GaN" "AlGaN" 0.1 1.8)

;(sdedr:define-refinement-placement "Ref.channel" "Ref.channel" "Pl.channel")







(sde:build-mesh "snmesh" "" "n@node@")








