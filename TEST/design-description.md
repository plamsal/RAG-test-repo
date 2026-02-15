<div align="center">

**Assignment 5: Design Description**

**Job Offer Comparison Application**

**Pratik Lamsal**  
Georgia Institute of Technology  
CS 6300 - Software Development Process  
February 15, 2026

</div>

---

## Requirement 1: Main Menu

When the app is started, the user is presented with the main menu, which allows the user to (1) enter or edit current job details, (2) enter job offers, (3) adjust the comparison settings, or (4) compare job offers (disabled if no job offers were entered yet).

**How is it realized:**

To realize this requirement, I created the `JobComparisonApp` class to serve as the main controller and entry point for the application. The `main()` method initializes the system and starts the application. The `displayMainMenu()` method presents the four menu choices from the requirement: (1) enter/edit current job, (2) enter job offers, (3) adjust comparison settings, and (4) compare job offers. Each menu choice maps to a corresponding method in JobComparisonApp: choice 1 is handled by `enterCurrentJobDetails()`, choice 2 by `startJobOfferEntry()` along with `saveJobOffer()` and `cancelJobOfferEntry()`, choice 3 by `adjustComparisonSettings()`, and choice 4 by `compareJobOffers()`. 

Additionally, the class maintains three key attributes: a `currentJob` attribute to store the user's current job (if entered), a `jobOffers` attribute (List<Job> with 0..* multiplicity, indicating zero or more offers can be stored) to store all entered job offers, and a `settings` attribute (of type ComparisonSettings) to store the user's comparison weight preferences.

To support the requirement that choice 4 should be disabled when no offers exist, I added a `hasJobOffers()` method that checks whether the `jobOffers` list is empty. The `displayMainMenu()` method calls `hasJobOffers()` to determine whether to enable or disable the compare option before presenting the menu to the user. The actual menu display is handled by the GUI, but the application logic determining menu state resides in JobComparisonApp.

---

## Requirement 2: Enter/Edit Current Job

**Requirement:**  

When choosing to enter current job details, a user will be shown a user interface to enter (if it is the first time) or edit all the details of their current job, which consists of: Title, Company, Location (city and state), Cost of living index, Yearly salary, Yearly bonus, Stock Option Shares, Wellness Stipend ($0-$1200), Life Insurance (0-10% of yearly salary), and Personal Development Fund ($0-$6000). The user can either save the job details or cancel and exit without saving, returning in both cases to the main menu.

**How is it realized:** 
To realize this requirement, the `enterCurrentJobDetails()` method in JobComparisonApp handles the workflow for entering or editing current job information. All job details are stored in the `Job` class, which contains attributes  for title, company, costOfLivingIndex, yearlySalary, yearlyBonus, stockOptionShares, wellnessStipend, lifeInsurancePercent, and personalDevelopmentFund. 

The location information, which includes the city and state, is encapsulated in a separate class called `Location`. The `Job` class has a composition relationship with this `Location` class, which is represented by the filled diamond and 1-to-1 relationship between the classes. This encapsulation separates the location information and makes the system more maintainable in case additional location information attributes are required in the future. The `Job` class also includes a boolean attribute called `isCurrentJob` to differentiate the current job from job offers while displaying ranked lists.

The `validate()` method in the Job class ensures all constraints are met: wellnessStipend must be between $0 and $1200, lifeInsurancePercent must be between 0 and 10, and personalDevelopmentFund must be between $0 and $6000. When the user saves the job, it is stored in the `currentJob` attribute of JobComparisonApp (with 0..1 multiplicity, indicating that a current job may or may not exist). If the user cancels, the changes are discarded and control returns to the main menu via `displayMainMenu()`. The actual UI presentation is handled by the GUI layer, while the data storage and validation logic resides in the Job and JobComparisonApp classes.

---
## Requirement 3: Enter Job Offers

When choosing to enter job offers, a user will be shown a user interface to enter all the details of the offer, which are the same ones listed above for the current job. The user can either save the job offer details or cancel. After saving, the user can (1) enter another offer, (2) return to the main menu, or (3) compare the offer (if they saved it) with the current job details (if present).

**How is it realized:**

To realize this requirement, the `startJobOfferEntry()` method in JobComparisonApp initiates the job offer entry process and creates a new Job object with empty values for the user to fill in. Job offers use the same Job class as the current job, ensuring structural consistency and reusing all the same attributes (title, company, location, salary, etc.) and validation logic. 

When the user completes entering the offer details, the `saveJobOffer(offer: Job)` method validates and adds the job offer to the `jobOffers` list in JobComparisonApp (with 0..* multiplicity, indicating zero or more offers can be stored). This aggregation relationship (hollow diamond) indicates that Job objects can exist independently and multiple offers can be maintained. If the user chooses to cancel instead, the `cancelJobOfferEntry()` method discards the draft offer without saving it, and control returns to the main menu.

After successfully saving an offer, the requirement states the user has three options. Option (1), entering another offer, simply calls `startJobOfferEntry()` again to begin a new entry cycle. Option (2), returning to the main menu, is handled by calling `displayMainMenu()`. Option (3), comparing the newly saved offer with the current job, is handled by the `compareOfferWithCurrentJob(offer: Job)` method, which provides immediate comparison between the specific offer that was just saved and the current job. This comparison option is only available if a current job exists, which is verified by the `hasCurrentJob()` method. The actual UI presentation and user selection of these three options is handled by the GUI layer, while the workflow logic resides in JobComparisonApp.

---

## Requirement 4: Adjust Comparison Settings

When adjusting the comparison settings, the user can assign integer weights to: Yearly salary, Yearly bonus, Stock Option Shares, Wellness Stipend, Life Insurance, and Personal Development Fund. These factors should be integer-based from 0 (no interest/don't care) to 9 (highest interest). Default value for all weights is 1. If no weights are assigned, all factors are considered equal. The user must be able to either save the comparison settings or cancel; both will return the user to the main menu.

**How is it realized:**

To realize this requirement, the `adjustComparisonSettings()` method in JobComparisonApp provides the interface for modifying weight values. All comparison weights are stored in the ComparisonSettings class, which contains six integer attributes: yearlySalaryWeight, yearlyBonusWeight, stockOptionSharesWeight, wellnessStipendWeight, lifeInsuranceWeight, and personalDevelopmentFundWeight. Each weight attribute has a constraint of {0..9} explicitly documented in the UML diagram to enforce the valid range specified in the requirement. All weights are initialized with a default value of 1, ensuring that if the user never adjusts settings, all compensation factors are weighted equally as required.

The ComparisonSettings class provides three supporting methods. The `validate()` method ensures all weight values remain within the valid range of 0 to 9 before saving. The `getTotalWeight()` method returns the sum of all six weights, which serves as the denominator in the weighted average calculations used for job scoring (as specified in Requirement 6). The `resetToDefaults()` method allows users to restore all weights back to the default value of 1, implementing the equal weighting behavior.

JobComparisonApp has a composition relationship with ComparisonSettings (indicated by the filled diamond and 1-to-1 multiplicity), meaning the settings object is owned by and exists only within the context of the application. When the user saves their weight adjustments, the changes are persisted in the settings object and control returns to the main menu via `displayMainMenu()`. If the user cancels, any modifications are discarded and the previous weight values remain unchanged, with control also returning to the main menu. The actual UI for entering weight values is handled by the GUI layer, while the data storage and validation logic resides in the ComparisonSettings class.

---

## Requirement 5: Compare Job Offers

When choosing to compare job offers, a user will: (a) be shown a list of job offers, displayed as Title and Company, ranked from best to worst, and including the current job (if present), clearly indicated; (b) select two jobs to compare and trigger the comparison; (c) be shown a table comparing the two jobs, displaying, for each job: Title, Company, Location, Yearly salary adjusted for cost of living, Yearly bonus adjusted for cost of living, Stock Option Shares (SOS), Wellness Stipend (WS), Life Insurance (LI), Personal Development Fund (PDF), and Job Score (JS); (d) be offered to perform another comparison or go back to the main menu.

**How is it realized:**

The `compareJobOffers()` method in JobComparisonApp orchestrates the comparison workflow. For requirement 5a, JobComparator's `rankJobs()` method calculates scores using `calculateJobScore()` and returns jobs sorted from best to worst. The `isCurrentJob` boolean attribute identifies the current job in the ranked list. 

For requirements 5b and 5c, the `compareJobs(job1, job2, settings)` method creates a ComparisonResult object containing both jobs and their scores. The ComparisonResult's `getJob1Data()` and `getJob2Data()` methods return Maps with all display data: title, company, location (via `getLocation()`), adjusted salary and bonus (via `getAdjustedYearlySalary()` and `getAdjustedYearlyBonus()`), other compensation attributes, and calculated scores.

For requirement 5d, `compareJobOffers()` allows repeating comparisons or returning to the main menu. JobComparisonApp has a dependency on JobComparator (dashed "uses" arrow), and JobComparator has a dependency on ComparisonResult (dashed "creates" arrow). The GUI handles display while JobComparator and ComparisonResult handle calculation and data packaging.

---

## Requirement 6: Job Score Calculation

When ranking jobs, a job's score is computed as the weighted average of: AYS + AYB + (SOS/3) + WS + (LI/100 * YS) + PDF, where AYS is yearly salary adjusted for cost of living, AYB is yearly bonus adjusted for cost of living, SOS is stock option shares, WS is wellness stipend, LI is life insurance percentage, YS is yearly salary (unadjusted), and PDF is personal development fund. For example, if the weights are 2, 2, 2, 1, 1, 1, the score would be: JS = 2/9 * AYS + 2/9 * AYB + 2/9 * (SOS/3) + 1/9 * WS + 1/9 * (LI/100 * YS) + 1/9 * PDF.

**How is it realized:**

The `calculateJobScore(job, settings)` method in JobComparator implements the weighted average formula. It uses the `getNormalizedWeights(settings)` helper method to calculate weight fractions (e.g., 2/9) by dividing each weight by the total from `ComparisonSettings.getTotalWeight()`. 

The method obtains AYS and AYB from `job.getAdjustedYearlySalary()` and `getAdjustedYearlyBonus()` (which divide by costOfLivingIndex and multiply by 100), divides SOS by 3 for the vesting period, accesses WS and PDF directly from Job attributes, and calculates the life insurance component using unadjusted salary. The final score sums all weighted components as shown in the requirement example. This centralized scoring logic in JobComparator is reused by both `rankJobs()` and `compareJobs()`, separating calculation from data storage.

---

## Requirement 7: User Interface

The user interface must be intuitive and responsive.

**How is it realized:**

This requirement is not represented in the UML design, as it will be handled entirely within the GUI implementation. The design focuses on application logic and data structures, while UI responsiveness and intuitiveness are presentation layer concerns that do not affect the class structure, relationships, or methods shown in the diagram.

---

# Additional Design Decisions

Location Class
Location information (city and state) is encapsulated in a separate class rather than storing these fields directly in the Job class. This improves modularity and helps to make it more maintaianable.

ComparisonResult Class
Comparison outputs are encapsulated in a dedicated ComparisonResult class to separate the concerns of data calculation (performed by JobComparator) from data presentation (handled by the GUI layer). This class encapsulates both jobs being compared along with their computed scores. The class also allows for retrieval of formatted display information through the methods `getJob1Data()` and `getJob2Data()`.

JobComparator Class
The JobComparator class is implemented as a stateless utility class with only static methods for scoring, ranking, and comparison operations. This follows good practice in object-oriented programming, which separates calculation logic from storage. The statelessness of the class also makes it easier to test the scoring algorithm in isolation. The class does not have any attributes, which aligns with the utility class pattern.

---

# Assumptions

1. **Cost of living index is always greater than zero** to prevent division by zero errors when calculating adjusted salary and bonus values.
2. **Monetary values are stored as double** to accommodate decimal precision for salary, bonus, and other compensation amounts.
3. **The GUI layer handles all user input validation** before passing data to the application logic layer, though backend validation via `validate()` methods provides additional integrity checks.
4. **Job offers and current job can be compared even if they have different cost of living indices**, as the adjustment formula normalizes these differences.


---

