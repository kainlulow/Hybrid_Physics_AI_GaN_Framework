File {
  Grid =    "@tdr@"
  Parameter = "@parameter@"
  Plot =   "@tdrdat@"
  Current = "@plot@"
  Output = "@log@"
}

## Electrical contacts
Electrode {
	
	{ Name="gate"   Voltage= (0 at 0 , @Vgstart@ at 3  @Vgend@ at 13 ) } #Schottky workfunction= 4.5}
	{ Name="source" Voltage= 0} # Resistence= 0.612}
	{ Name="drain"  Voltage= (0 at 0, -5 at 3  -5 at 20) } #  Resistence= 0.612}
	
}

## Physics models
Physics {
	 AreaFactor= 1000
		eQuantumPotential(density) hQuantumPotential(density)
	Mobility (	  
	  DopingDependence
          
	  HighFieldSaturation
          Enormal(
          Lombardi_highk )

	)
	Fermi
	EffectiveIntrinsicDensity(Slotboom)
        Thermionic
#if @Strain@ == 1
	Piezoelectric_Polarization (strain)
#endif
    Recombination(SRH Auger)

}
 

 

#if @Strain@ == 1
Physics (MaterialInterface="Al2O3/InGaN"){
       PiezoElectric_Polarization(activation= 0)
}

#if @AlN1@ == 1
Physics (MaterialInterface="InGaN/AlN") {
	PiezoElectric_Polarization(activation= @InGaN_AlN@) #0.3)
}

Physics (MaterialInterface="AlN/AlGaN") {
	PiezoElectric_Polarization(activation= @AlN_AlGaN@) #0.9)
}
#else
Physics (MaterialInterface="InGaN/AlGaN") {
	PiezoElectric_Polarization(activation= @InGaN_AlGaN@) #0.3)
}
#endif

#if @AlN2@ == 1 & @AlN1@ == 0 
Physics (MaterialInterface="AlGaN/AlN") {
	PiezoElectric_Polarization(activation= @AlN_AlGaN@) #0.3)
}
#endif
#if @AlN2@ == 1 
Physics (MaterialInterface="AlN/GaN") {
	PiezoElectric_Polarization(activation= @AlN_GaN@) #0.3)
}

#else
Physics (MaterialInterface="AlGaN/GaN") {
	PiezoElectric_Polarization(activation= @AlGaN_GaN@) #0.3)
}
#endif

Physics (MaterialInterface="Silicon/GaN") {
	PiezoElectric_Polarization(activation= 0)
}
#endif


#Physics (Region="GaNb") {
#	Traps(
#		(Acceptor Level Conc=5e14 EnergyMid=0.45 FromCondBand)
#	)
#}




Math {
	Transient=BE
	ExitOnFailure
	Extrapolate
	Iterations= 15
	Digits= 8
	RHSMin= 1e-20
    RHSMax= 1e30
	eDrForceRefDens= 1e8 hDrForceRefDens= 1e8
	Derivatives
	 method= ILS(set=12)
        ILSrc= "set (12) { 
		iterative (gmres(100), tolrel=1e-9,  tolunprec=1e-4, tolabs=0, maxit=250);
		preconditioning(ilut(1e-8,-1),left);
		ordering(symmetric=nd, nonsymmetric=mpsils);
		options( compact=yes, linscale=0, verbose=0, refineresidual=20);
	}; "
	ExtendedPrecision(128)
	NumberOfThreads= 8
	 	NewtonPlot (
		Error MinError
		Residual
)


	
}



Plot {
  Potential eQuasiFermi hQuasiFermi
  eDensity hDensity SpaceCharge
  Current eCurrent hCurrent CurrentPotential
  ElectricField SemiconductorElectricField InsulatorElectricField
  eMobility hMobility
  eVelocity hVelocity
  Doping DonorConcentration AcceptorConcentration
  ElectricField/vector
 
  PE_Polarization/vector
  PE_Charge PiezoCharge
  ConductionBandEnergy ValenceBandEnergy
  ElectronAffinity
  BandGap EffectiveBandGap BandGapNarrowing
  QCEffectiveBandGap
  xMolefraction yMolefraction
  LatticeTemperature eTemperature hTemperature
  lHeatFlux TotalHeat
  eJouleHeat hJouleHeat JouleHeat
  PeltierHeat ThomsonHeat RecombinationHeat
  DissipationRateDensity
  OpticalAbsorptionHeat
  ThermalConductivity LatticeHeatCapacity
}

Solve {Coupled (Iterations= 10000 LinesearchDamping= 1e-6) {Poisson}
	Coupled (Iterations= 10000 LinesearchDamping= 1e-6) {Poisson Hole}
	Coupled (Iterations= 10000 LinesearchDamping= 1e-6) {Poisson Electron Hole}	
	Plot(Fileprefix="n_n@node@_equilibrium")	

#if @EquiOnly@ != 1	
* -- Ramp Vd from 0 to -3V in 3 sec.--
   Transient (
       InitialTime= 0 InitialStep= 0.02 Increment= 1.5 Decrement= 2.0
       Maxstep= 0.1 FinalTime= 3
    ) { coupled { Poisson Electron Hole } }
   
    NewCurrentFile="IdVg_" 
* -- Ramp Vg and vd from 0 to -5 V in 10 sec (from 3 to 13).--
   Transient (
       InitialTime= 3 InitialStep= 0.02 Increment= 1.5 Decrement= 2.0
       Maxstep= 0.10 FinalTime= 13 
    )  {coupled{Poisson Electron Hole } 
    
    Plot(-Loadable Fileprefix="n_n@node@_inter" NoOverWrite Time=(Range=(3 13) Intervals=@ntdr@))}	
#endif
    		
}


