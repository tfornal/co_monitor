# The C/O monitor code for comprehensive data analysis

The code was created to perform various calculations of potential C/O monitor signals and validate experimental data of the Lyman-alpha lines measurements of hydrogen like ions of B, C, N and O. The code calculates only the line intensities of the given transitions but does not calculate the background radiation (at least not for now).

## Things TODO:
- setup!!! i caly szajs do automatycznego uruchamiania
- zaimplementowac w simulations protection shields - aby moc wyliczac transmisje; 
- sprawdzic wspolrzedne ecrh shields - mozliwe przesuniecie w gornym albo dolnejj komorze
- move predefined profiles, twogauss sum etc to external module - so called helper; 
- YAML!!!!!!!!! or TOML??? jako config ini?
- improve documentation
- external file with all settings (.dat or something else;)
- optimalization; - multithreading, multiprocessing;
- closure of the B and N collimators from both sides - does not converge - check the reason - possibly due to the wrong input data for the bottom chamber?
- implementation of CXRS transition - checkout of max Reff etc;
- implementation of one stanard of input files;
- TESTy jednostkowe i integracyjne (pokrycie minimum 50%)
- jeden meta plik z poziomu ktorego uruchamiam różne modułyu (1 z 3, albo wszystkie po kolei)
- implementation of logs;
- implementation of crystal rocking curves;
- checking up of the lines occuring close to the investigated lyman alpha transitions 
- recalculation of all geometry for standard confifuration; 
- implementation of remote desktop on other machine and perform calculations on request; 
- implementation of additional lines which are bounded together with considered lyman-alpha transitions;
- implementation of SQL or other database for faster data validation;
- parallelization of various python methods;
- implementation of background radiation????
- implementation of static typing!!!!!!!!!
-
- machine learning / Bayes (?) for fast data processing - once some experimental data will be obtained

## The code consists of three main sections:
### Module 1 - Reff calculation
Using VMEC code. Field line tracer will be included for edge plasma modeling.
Development of the description TBD

### Module 2 - Geometry calculation
Development of the description TBD 

### Module 3 - Emissivity calculation
Development of the description TBD