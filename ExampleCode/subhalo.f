      PROGRAM subhalo
c-----------------------------------------------------------------------
c
c-----------------------------------------------------------------------
       
      IMPLICIT NONE

      INTEGER   Nmax
      PARAMETER (Nmax=100000000)
      
      INTEGER   i,j,k,l,kk,isim,igroup,iversion,Nhis
      INTEGER*8 ID,PID,UID 
      INTEGER   IDsmall,Nhalo,Nhost,Nsub
      INTEGER   IDhalo(Nmax),iFlag(Nmax),revID(Nmax)
      REAL      msub,Mhost,Vvir,mrat,vrat,bin,xMin,xMax
      REAL      dNdlogM,dNdlogMrat,dNdlogVrat,Volume,Mmin,MratMin
      REAL      Mvir,Rvir,Rs,Vmax,Macc,Mpeak,Vacc,Vpeak
      REAL      haloMvir(Nmax),haloVmax(Nmax),his(200,2),his2(200)
      CHARACTER*60  infile,outfile,outfil0,outfil1,outfil2,outfil3,
     &      outfil4,outfil5,outfil6,outfil7,outfil8,outfil9,outfil10
        
c-----------------------------------------------------------------------
        
c---read mass limits of host halos

      WRITE(*,*)' Which simulation? (1=Bolshoi, 2=MultiDark)'
      READ(*,*)isim
      WRITE(*,*)' '
      
      WRITE(*,*)' Which group finder? (1=Rockstar, 2=BDM)'
      READ(*,*)igroup
      WRITE(*,*)
      
      IF (isim.EQ.1.AND.igroup.EQ.1) iversion=1
      IF (isim.EQ.1.AND.igroup.EQ.2) iversion=2
      IF (isim.EQ.2.AND.igroup.EQ.1) iversion=3
      IF (isim.EQ.2.AND.igroup.EQ.2) iversion=4
      
      WRITE(*,*)' given logarithmic bin width '
      READ(*,*)bin
      WRITE(*,*)' '
      
      Nhis = INT(6.0/bin) + 1
      IF (Nhis.GT.200) THEN
        WRITE(*,*)'binwidth too small'
        STOP
      END IF

c---set filenames, simulation volume, and minimum halo mass, 
c   corresponding to 100 particles

      IF (iversion.EQ.1) THEN
        infile  ='HaloCat_Bolshoi_Rockstar_z0.dat'
        outfil0 ='dNdM_Bolshoi_Rockstar.dat'
        outfil1 ='SHMF_Bolshoi_Rockstar_138_140.dat'
        outfil2 ='SHMF_Bolshoi_Rockstar_136_138.dat'
        outfil3 ='SHMF_Bolshoi_Rockstar_134_136.dat'
        outfil4 ='SHMF_Bolshoi_Rockstar_132_134.dat'
        outfil5 ='SHMF_Bolshoi_Rockstar_130_132.dat'
        outfil6 ='SHMF_Bolshoi_Rockstar_128_130.dat'
        outfil7 ='SHMF_Bolshoi_Rockstar_126_128.dat'
        outfil8 ='SHMF_Bolshoi_Rockstar_124_126.dat'
        outfil9 ='SHMF_Bolshoi_Rockstar_122_124.dat'
        outfil10='SHMF_Bolshoi_Rockstar_120_122.dat'
        Volume = 250.0**3
        Mmin = 1.35E+10
      END IF

      IF (iversion.EQ.2) THEN
        infile  ='HaloCat_Bolshoi_BDM_z0.dat'
        outfil0 ='dNdM_Bolshoi_BDM.dat'
        outfil1 ='SHMF_Bolshoi_BDM_138_140.dat'
        outfil2 ='SHMF_Bolshoi_BDM_136_138.dat'
        outfil3 ='SHMF_Bolshoi_BDM_134_136.dat'
        outfil4 ='SHMF_Bolshoi_BDM_132_134.dat'
        outfil5 ='SHMF_Bolshoi_BDM_130_132.dat'
        outfil6 ='SHMF_Bolshoi_BDM_128_130.dat'
        outfil7 ='SHMF_Bolshoi_BDM_126_128.dat'
        outfil8 ='SHMF_Bolshoi_BDM_124_126.dat'
        outfil9 ='SHMF_Bolshoi_BDM_122_124.dat'
        outfil10='SHMF_Bolshoi_BDM_120_122.dat'
        Volume = 250.0**3
        Mmin = 1.35E+10
      END IF

      IF (iversion.EQ.3) THEN
        infile  ='HaloCat_MultiDark_Rockstar_z0.dat'
        outfil0 ='dNdM_MultiDark_Rockstar.dat'
        outfil1 ='SHMF_MultiDark_Rockstar_148_150.dat'
        outfil2 ='SHMF_MultiDark_Rockstar_146_148.dat'
        outfil3 ='SHMF_MultiDark_Rockstar_144_146.dat'
        outfil4 ='SHMF_MultiDark_Rockstar_142_144.dat'
        outfil5 ='SHMF_MultiDark_Rockstar_140_142.dat'
        outfil6 ='SHMF_MultiDark_Rockstar_138_140.dat'
        outfil7 ='SHMF_MultiDark_Rockstar_136_138.dat'
        outfil8 ='SHMF_MultiDark_Rockstar_134_136.dat'
        outfil9 ='SHMF_MultiDark_Rockstar_132_134.dat'
        outfil10='SHMF_MultiDark_Rockstar_130_132.dat'
        Volume = 1000.0**3
        Mmin = 8.64E+11
      END IF

      IF (iversion.EQ.4) THEN
        infile  ='HaloCat_MultiDark_BDM_z0.dat'
        outfil0 ='dNdM_MultiDark_BDM.dat'
        outfil1 ='SHMF_MultiDark_BDM_148_150.dat'
        outfil2 ='SHMF_MultiDark_BDM_146_148.dat'
        outfil3 ='SHMF_MultiDark_BDM_144_146.dat'
        outfil4 ='SHMF_MultiDark_BDM_142_144.dat'
        outfil5 ='SHMF_MultiDark_BDM_140_142.dat'
        outfil6 ='SHMF_MultiDark_BDM_138_140.dat'
        outfil7 ='SHMF_MultiDark_BDM_136_138.dat'
        outfil8 ='SHMF_MultiDark_BDM_134_136.dat'
        outfil9 ='SHMF_MultiDark_BDM_132_134.dat'
        outfil10='SHMF_MultiDark_BDM_130_132.dat'
        Volume = 1000.0**3
        Mmin = 1.2E+12
      END IF
                 
c---initialize

      DO i=1,Nmax
        revID(i)=0
        IDhalo(i)=0
        iFlag(i) =0
        haloMvir(i) = 0.0
        haloVmax(i) = 0.0
      END DO
                          
      DO i=1,200
        his(i,1) =0.0
        his(i,2) =0.0
        his2(i)=0.0
      END DO

c---read data
            
      OPEN(20,file=infile,status='OLD')

      Nhalo = 0
      DO WHILE (.TRUE.)
        IF (iversion.EQ.4) THEN
          READ(20,98,end=100)j,ID,Mvir,Vmax,UID
        ELSE   
          READ(20,99,end=100)j,ID,PID,UID,Mvir,Rvir,Rs,Vmax,Macc,
     &                     Mpeak,Vacc,Vpeak
        END IF
        
        Nhalo = Nhalo + 1 
        IF (Nhalo.NE.j) THEN
          WRITE(*,*)' Error reading data',Nhalo,j
          STOP
        END IF   

        IDsmall = MOD(ID,10000000)
        
        IDhalo(j) = IDsmall
        IF (UID.LT.0) THEN
          iFlag(j) = UID
        ELSE   
          iFlag(j) = MOD(UID,10000000)
        END IF 
        haloMvir(j) = Mvir
        haloVmax(j) = Vmax
        revID(IDsmall) = j

c        WRITE(*,*)j,ID,IDhalo(j),iFlag(j),haloMvir(j)
c        IF (Nhalo.GE.100) STOP 
        IF (IDhalo(j).GT.Nmax.OR.iFlag(j).GT.Nmax.OR.j.GT.Nmax) THEN
          WRITE(*,*)' Memory Overflow: ',j,IDhalo(j),iFlag(j),Nmax
        END IF 

      END DO
 100  CONTINUE 
      CLOSE(20)      

      WRITE(*,*)' Total number of haloes read: ',Nhalo
      WRITE(*,*)' '
       
c---compute halo mass function and write to file

      DO j=1,Nhalo
        IF (iFlag(j).LT.0) THEN
          Mhost= haloMvir(j)
          kk = INT((Mhost - 10.0)/0.2) + 1
          his2(kk) = his2(kk) + 1.0
        END IF
      END DO

      OPEN(11,file=outfil0,status='UNKNOWN') 
      DO kk=1,30
        Mhost = 10.0 + FLOAT(kk-1) * 0.2 + 0.1
        dNdlogM = ALOG10(MAX(his2(kk)/0.2/Volume,1.0E-20))
        IF (Mhost.GE.ALOG10(Mmin)) WRITE(11,91)kk,Mhost,dNdlogM
      END DO  
      CLOSE(11)

c---compute SHMF for 10 host halo mass bins

      DO l=1,10
      
        DO i=1,200
          DO j=1,2  
            his(i,j)=0.0
          END DO
        END DO

        IF (isim.EQ.2) THEN
          xMin = 14.8 - FLOAT(l-1)*0.2
          xMax = xMin + 0.2
        ELSE
          xMin = 13.8 - FLOAT(l-1)*0.2
          xMax = xMin + 0.2
        END IF  
        MratMin = Mmin/(10.0**((xMax + xMin)/2.0))

        Nsub = 0
        Nhost = 0
        DO j=1,Nhalo
          IF (iFlag(j).GT.0) THEN
            msub = haloMvir(j)
            Vmax = haloVmax(j)
            IF (revID(iFlag(j)).EQ.0) THEN
ccc              WRITE(*,*)' ALARM ',j,iFlag(j),revID(iFlag(j)),msub
              Mhost = 0.0
            ELSE
              Mhost = haloMvir(revID(iFlag(j)))
              Vvir = 2.15864 + (Mhost-12.0)/3.0
            END IF
            IF (Mhost.GE.xMin.AND.Mhost.LT.xMax) THEN
              mrat = msub-Mhost
              vrat = ALOG10(Vmax) - Vvir
              Nsub = Nsub + 1
              IF (mrat.GE.-6.0.AND.mrat.LT.0.0) THEN
                k =  INT((mrat + 6.0)/bin) + 1
                his(k,1) = his(k,1) + 1.0
              END IF
              IF (vrat.GE.-2.0.AND.vrat.LT.0.0) THEN
                k =  INT((vrat + 2.0)/(bin/3.0)) + 1
                his(k,2) = his(k,2) + 1.0
              END IF
            END IF  
          ELSE    
            Mhost= haloMvir(j)
            IF (Mhost.GE.xMin.AND.Mhost.LT.xMax) Nhost = Nhost+1
          END IF
        END DO  
          
        WRITE(*,88)l,xMin,xMax,Nhost,Nsub

        IF (l.EQ.1) outfile=outfil1
        IF (l.EQ.2) outfile=outfil2
        IF (l.EQ.3) outfile=outfil3
        IF (l.EQ.4) outfile=outfil4
        IF (l.EQ.5) outfile=outfil5
        IF (l.EQ.6) outfile=outfil6
        IF (l.EQ.7) outfile=outfil7
        IF (l.EQ.8) outfile=outfil8
        IF (l.EQ.9) outfile=outfil9
        IF (l.EQ.10) outfile=outfil10
                    
        OPEN(10,file=outfile,status='UNKNOWN') 
        DO k=1,Nhis
          mrat = -6.0 + FLOAT(k-1) * bin + (bin/2.0)
          vrat = -2.0 + FLOAT(k-1) * (bin/3.0) + (bin/6.0)
          dNdlogMrat = his(k,1)/FLOAT(Nhost)/bin
          dNdlogMrat = MAX(1.0E-20,dNdlogMrat)
          dNdlogVrat = his(k,2)/FLOAT(Nhost)/(bin/3.0)
          dNdlogVrat = MAX(1.0E-20,dNdlogVrat)
          IF (mrat.GE.ALOG10(MratMin)) THEN
           WRITE(10,90)k,mrat,ALOG10(dNdlogMrat),INT(his(k,1)),
     &                   vrat,ALOG10(dNdlogVrat),INT(his(k,2))
          END IF 
        END DO  
        CLOSE(10)
         
      END DO

88    FORMAT(I2,2X,F5.2,2X,F5.2,2X,I9,2X,I9)
90    FORMAT(I4,2X,2(F6.2,2X,F9.4,2X,I8))
91    FORMAT(I4,2X,F6.3,2X,F8.4)
98    FORMAT(I8,2X,I8,2X,F7.4,2X,F6.1,2X,I9)
99    FORMAT(I8,I12,2X,I12,2X,I12,2X,F7.4,3(2X,F8.3),2X,F7.4,
     &2X,F7.4,2X,F8.3,2X,F8.3)  


      STOP
      END 
