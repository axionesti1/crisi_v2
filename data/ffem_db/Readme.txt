Title: 			FFEM-DB Database of Flood Fatalities from the Euro-Mediterranean region
##########################################################################################################################################################################

Authors: 				Petrucci, O. (Olga)||orcid:0000-0001-6918-1135
					Mercuri, M. (Michele)||orcid:0000-0002-5217-6644
					Aceto, L. (Luigi)
					Bianchi, C. (Cinzia)
					Brázdil, R. (Rudolf)||orcid:0000-0003-4826-2299
					Diakakis, M. (Michalis)||orcid:0000-0001-6155-6588
					Inbar, M. (Moshe)
					Kahraman, A. (Abdullah)||orcid:0000-0002-8180-1103
					Kılıç, O. (Özgenur)
					Krahn, A. (Astrid)
					Kreibich, H. (Heidi)||orcid:0000-0001-6274-3625
					Kotroni, V. (Vassiliki)||orcid:0000-0003-1248-5490
					De Brito, M.M. (Mariana Madruga)||orcid:0000-0003-4191-1647
					Llasat, M.C. (Maria Carmen)||orcid:0000-0001-8720-4193
					Llasat-Botija, M. (Montserrat)
					Macdonald, N. (Neil)||orcid: 0000-0003-0350-7096
					Papagiannaki, K. (Katerina)||orcid:0000-0003-4433-5841
					Pereira, S. (Susana)||orcid:0000-0002-9674-0964
					Řehoř, J. (Jan)
					Rossello-Geli, J. (Joan)||orcid:0000-0002-5299-7039
					Salvati, P. (Paola)||orcid:0000-0002-4305-2105
					Vinet, F. (Freddy)
					Zêzere, J.L. (José Luis)||orcid:0000-0002-3953-673X
###################################################################################################################################################################################

Related Publications: 			https://www.nature.com/articles/s41597-022-01273-x#citeas
                                        https://www.mdpi.com/2073-4441/11/8/1682
					https://doi.org/10.4121/uuid:3ac47b36-12ce-4c38-97d5-5bbf7d20572d
					https://doi.org/10.4121/uuid:489d8a13-1075-4d2f-accb-db7790e4542f
					https://doi.org/10.4121/14754999.v1

###################################################################################################################################################################################

Description:				Description: FFEM-DB (Database of Flood Fatalities from the Euro-Mediterranean region) is a database which contains 2.875 cases of flood
					fatalities that occurred throughout 41 years (1980–2020) in 12 study areas in Europe (Cyprus; Czech Republic; Germany; Greece; Israel; Italy; Portugal;
					Turkey; United Kingdom; the Spanish regions of Balearic Islands and Catalonia, and the Mediterranean regions of South France).
					FFEM-DB provides not only the number of fatalities, but also detailed information about the profile of victims and the circumstances
					of the accidents. Flood fatality cases are georeferenced using NUTS 3 level (Nomenclature of Territorial Units for Statistics),
					allowing analyses of fatality distribution in respect to geographic and demographic data.
					FFEM-DB data are stored in a relational database made of three tables: 
					1. FATALITIES. This table contains the date of the fatal accident, victim’s profile (gender, age, and residency)
					and the circumstances of the fatal accident (victim condition and activity, accident place and dynamic, death cause
					and either protective or hazardous behaviors). The ID-Fatality is the primary key connecting this table to the table LOCATION
					while NUTS_3_ID works as a foreign key for the connection to the table NUTS 3.
					2. LOCATION. The table contains the details of the place where the accident occurred (country, territorial levels from 1 to 3
					(according to country administrative subdivisions), latitude and longitude, accuracy of localization and the name and acronym
					of the study area). FATALITY_ID is the primary key, and the NUTS_3_ID works as a foreign key to connect this table to the table NUTS 3.
					3. NUTS 3. This table allows the downscaling of accident place from the country level to NUTS 3 level.
					NUTS (Nomenclature of Territorial Units for Statistics, Eurostat. Version 1/2/2020) is a European four levels hierarchical classification
					that subdivides each Member State (identified as NUTS 0 level) into a number of NUTS 1 regions each of which is in turn subdivided into a
					number of NUTS 2 regions, and NUTS 3.
					In FFEM-DB, the table NUTS 3 contains all the European NUTS 3, not simply those where flood fatalities occurred, in the light of the possible future
					inclusion of other countries. Each NUTS 3 has its ID and name, the related NUTS 2, NUTS 1 and NUTS 0 IDs and names, area (NUTS_3_AREA, SqKm),
					population (NUTS_3_POPULATION) population density (NUTS_3_POPULATION DENSITY, inhabitants/SqKm), and population sorted by
					gender (NUTS_3_MALES NUTS_3_FEMALES), and age classes of males (NUTS_3_AGE_0-14_MAL, NUTS_3_AGE_15-29_MAL, NUTS_3_AGE_30-49_MAL,
					NUTS_3_AGE_50-64_MAL, and NUTS_3_AGE_OVER_64_MAL) and females (NUTS_3_AGE_0-14_FEM, NUTS_3_AGE_15-29_FEM, NUTS_3_AGE_30-49_FEM,
					NUTS_3_AGE_50-64-FEM, and NUTS_3_AGE_OVER_64_FEM).
					All data are updated to 2019 (source is Eurostat for all but ISR, for which data source is http://Cbs.gov.il-st02_03.xls).
					The field NUTS_3_ID acts as the primary key of the NUTS 3 table, and as a foreign key of the tables FATALITIES and LOCATION.


####################################################################################################################################################################################

Records: 				n.2875 records. - Each record contains a Flood Fatality.

####################################################################################################################################################################################



Description of Tables:
						 			     *******
									     ** 1 **
									****************
									** FATALITIES **
******************************************************************************************************************************************************************

FATALITY_ID (int – Primary Key):	| A unique, incremental number as identifier for each flood fatality.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_ID (varchar – Foreign Key):	| NUT 3 ID of the flood event.
________________________________________|_________________________________________________________________________________________________________________________
DATE (date):				| Date of the flood event (dd/mm/yyyy).
________________________________________|_________________________________________________________________________________________________________________________
AGE_STRING (enum):			| Age class of the flood fatality:
					|
					| 				from 0 to 14 years	Child
      					| 				from 15 to 29 years	Boy/Girl
   					| 				from 30 to 49 years	Young adult
       					| 				from 50 to 64 years	Adult
       					| 				>=64 years		Elderly
					|							Unknown
________________________________________|_________________________________________________________________________________________________________________________			
GENDER (enum):				| Gender of the flood fatality (M: male; F: female; Unknown).
________________________________________|_________________________________________________________________________________________________________________________
RESIDENCY (enum):			| Residency of flood fatality:
					|
                                        |				Resident
                                        |				Not resident
					|				Tourist
					|				Unknown
________________________________________|_________________________________________________________________________________________________________________________
VICTIM_CONDITION (enum):		| Victim condition at the time of the flood event:
					|
					| 						By bicycle	
      					| 						By boat
   					| 						By bus
       					| 						By car
       					| 						By caravan
					| 						By tractor	
      					| 						By truck
   					| 						By van
       					| 						Laying
       					| 						Standing
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
VICTIM_ACTIVITY	(enum):			| Victim activity at the time of the flood event:
					|
					| 						Travelling	
      					| 						Recreational activities
   					| 						Rescuing someone
       					| 						Sleeping
       					| 						Working
					| 						Hunting	
      					| 						Fishing
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
ACCIDENT_PLACE (enum):			| Place where the person was affected by the flood event:
					|
					| 						Public/private building	
      					| 						Bridge
   					| 						Campsite/tent
       					| 						Riverbed/riverside
       					| 						Tunnel/underpass
					| 						Countryside	
      					| 						Ford
   					| 						Recreation area
       					| 						Road
       					| 						Bungalow
					|						Unknown
________________________________________|________________________________________________________________________________________________________________________
ACCIDENT_DYNAMIC (enum):		| Dynamic of the accident.
					|
					| 						Blocked in a flooded room
      					| 						Caught in a bridge collapse
   					| 						Caught in a road collapse
       					| 						Caught in a building collapse
       					| 						Dragged by water/mud
					| 						Fallen into the river
      					| 						Surrounded by water/mud
   					| 						Hit
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
DEATH_CAUSE (enum):			| Death cause of the flood fatality:
					|
					| 						Collapse/heart attack
      					| 						Drowning
   					| 						Hypothermia
       					| 						Electrocution
       					| 						Poly-trauma
					| 						Poly-trauma and suffocation
      					| 						Suffocation
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
PROTECTIVE_BEHAVIOUR (enum):		| Protective behaviours of the flood fatality:
					|
					| 						Climbing trees
					|						Driving to avoid danger
					|						Getting on roof/upper floor
					|						Getting out of car
					|						Getting out of buildings
					|						Grabbing on to someone/something
					|						Moving to safer place
					|						Getting on the car roof
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
HAZARDOUS_BEHAVIOUR (enum):		| Hazardous behaviours of the flood fatality:
					|
					|						Check damage during flood
					|						Driving on roads closed by police
					|						Fording rivers
					|						Refuse evacuation
					|						Trying to rescue animals
					|						Refuse warnings
					|						Staying on bridges
					|						Staying on river banks
					|						Trying to save vehicles
					|						Trying to save belongings
					|						Unknown
________________________________________|_________________________________________________________________________________________________________________________
##################################################################################################################################################################
		

						  			      *******
						 			      ** 2 **
									******************
									**   LOCATION   **
******************************************************************************************************************************************************************

FATALITY_ID (int – Primary Key):	| A unique, incremental number as identifier for each flood fatality.
________________________________________|_________________________________________________________________________________________________________________________
COUNTRY (varchar):			| Country where the flood fatality occurred
________________________________________|_________________________________________________________________________________________________________________________
FFEM_STUDY_AREA (varchar):		| FFEM-DB study area where the flood fatality occurred:
					|
                                        |					Balearic Islands
                                        |					Catalonia
                                        |					Cyprus
					|					Czech Republic
					|					South France
					|					Germany
					|					Greece
					|					Israel
					|					Italy
					|					Portugal
					|					Turkey
					|					United Kingdom
________________________________________|_________________________________________________________________________________________________________________________
STUDY_AREA_ACHRONIM (varchar):		| String code of the countries where the flood event occurred:
					|
                                        |					BAL  for Balearic Islands
                                        |					CAT  for Catalonia
                                        |					CYP  for Cyprus
					|					CZE  for Czech Republic
					|					SFR  for South France
					|					GER  for Germany
					|					GRE  for Greece
					|					ISR  for Israel
					|					ITA  for Italy
					|					POR  for Portugal
					|					TUR  for Turkey
					|					UK  for United Kingdom
________________________________________|_________________________________________________________________________________________________________________________
TERRITORIAL_LV1 (varchar):		| Territorial level 1, according to country administrative subdivisions (i.e. Region).
________________________________________|_________________________________________________________________________________________________________________________
TERRITORIAL_LV2 (varchar):		| Territorial level 2, according to country administrative subdivisions (i.e. Municipality)
________________________________________|_________________________________________________________________________________________________________________________
TERRITORIAL_LV3 (varchar):		| Territorial level 3, according to country administrative subdivisions (i.e. Prefecture)
________________________________________|_________________________________________________________________________________________________________________________
LATITUDE (decimal):			| Latitude where the flood fatality occurred; Projection in WGS84 (EPSG: 4326).
________________________________________|_________________________________________________________________________________________________________________________
LONGITUDE (decimal):			| Longitude where the flood fatality occurred; Projection in WGS84 (EPSG: 4326).
________________________________________|_________________________________________________________________________________________________________________________
LOC_ACCURACY (enum):			| Accuracy of the coordinates (Low or High).
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_ID (varchar – Foreign Key):	| NUTS 3 ID where the flood fatality occurred.
________________________________________|__________________________________________________________________________________________________________________________
###################################################################################################################################################################


						     				*******
						     				** 3 **
									  ******************
									  **    NUTS 3    **
******************************************************************************************************************************************************************

NUTS_3_ID (varchar – Foreign Key):	| NUTS 3 ID.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_NAME (varchar):			| NUTS 3 Name.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_2_ID (varchar):			| NUTS 2 ID.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_2_NAME (varchar):			| NUTS 2 Name.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_1_ID (varchar):			| NUTS 1 ID.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_1_NAME (varchar):			| NUTS 1 Name.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_0_ID (varchar):			| NUTS 0 ID.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_0_NAME (varchar):			| NUTS 0 Name.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_AREA (decimal):			| NUTS 3 area [SqKm].
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_POPULATION (int):		| NUTS 3 population number.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_POP_DENSITY (decimal):		| NUTS 3 population density [inhabitants/SqKm].
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_MALES (int):			| NUTS 3 Male population.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_FEMALES (int):			| NUTS 3 Female population.
________________________________________|_________________________________________________________________________________________________________________________
NUTS_3_AGE_0-14_MAL (int):		|
________________________________________|
NUTS_3_AGE_0-14_FEM (int):		| 
________________________________________|
NUTS_3_AGE_15-29_MAL (int):		|
________________________________________|
NUTS_3_AGE_15-29_FEM (int):		|
________________________________________|
NUTS_3_AGE_30-49_MAL (int):		| 
________________________________________|		
NUTS_3_AGE_30-49_FEM (int):		| 		Population sorted by males and females and in age classes
________________________________________|
NUTS_3_AGE_50-64_MAL (int):		|
________________________________________|
NUTS_3_AGE_50-64_FEM (int):		|
________________________________________|
NUTS_3_AGE_OVER_64_MAL (int):		|
________________________________________|
NUTS_3_AGE_OVER_64_FEM (int):		|
________________________________________|
POP_AGE_NOTE (varchar):			|
________________________________________|__________________________________________________________________________________________________________________________
###################################################################################################################################################################

#END#