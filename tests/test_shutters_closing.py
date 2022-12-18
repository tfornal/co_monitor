import numpy as np
import matplotlib.pyplot as plt


### TODO - implement one meta class

"""C"""

slits_number = np.arange(0, 11)

intensities_top_gauss_profiles = np.array(
    [
        0,
        4.49e09,
        1.21e10,
        2.20e10,
        3.43e10,
        4.62e10,
        5.80e10,
        6.57e10,
        6.88e10,
        6.88e10,
        6.88e10,
    ]
)
intensities_bottom_gauss_profiles = np.array(
    [
        0,
        0.00e00,
        8.26e06,
        3.21e09,
        1.07e10,
        2.25e10,
        3.45e10,
        4.68e10,
        5.67e10,
        6.43e10,
        6.88e10,
    ]
)

intensity_top_predefined_profile_5 = np.array(
    [
        0,
        1.45e08,
        3.90e08,
        7.10e08,
        1.11e09,
        1.50e09,
        1.89e09,
        2.13e09,
        2.24e09,
        2.24e09,
        2.24e09,
    ]
)
intensity_bottom_predefined_profile_5 = np.array(
    [
        0,
        0.00e00,
        2.39e05,
        1.09e08,
        3.54e08,
        7.37e08,
        1.13e09,
        1.53e09,
        1.85e09,
        2.10e09,
        2.24e09,
    ]
)


######### TODO zrobic labele w plotach
plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n CVI sum_two_gauss_profiles"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(slits_number, intensities_top_gauss_profiles, label="top_to_bottom opening")
plt.plot(slits_number, intensities_bottom_gauss_profiles, label="bottom_to_top opening")
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n CVI predefined_profile"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(
    slits_number, intensity_top_predefined_profile_5, label="top_to_bottom opening"
)
plt.plot(
    slits_number, intensity_bottom_predefined_profile_5, label="bottom_to_top opening"
)
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()

print("CVI relative transmission change: ")
for j, i in enumerate(intensities_top_gauss_profiles):
    print(j + 1, "slit:", int(i / intensities_top_gauss_profiles[-1] * 100), "%")

print("CVI relative transmission change: ")
for j, i in enumerate(intensities_bottom_gauss_profiles):
    print(j + 1, "slit:", int(i / intensities_bottom_gauss_profiles[-1] * 100), "%")


"""O"""

intensities_top_gauss_profiles = np.array(
    [
        0,
        1.15e09,
        7.96e09,
        2.44e10,
        5.51e10,
        9.92e10,
        1.47e11,
        1.90e11,
        2.20e11,
        2.35e11,
        2.41e11,
    ]
)
intensities_bottom_gauss_profiles = np.array(
    [
        0,
        5.82e09,
        2.05e10,
        5.02e10,
        9.36e10,
        1.41e11,
        1.85e11,
        2.16e11,
        2.32e11,
        2.39e11,
        2.40e11,
    ]
)

intensity_top_predefined_profile_5 = np.array(
    [
        0,
        5.51e07,
        3.85e08,
        1.18e09,
        2.66e09,
        4.79e09,
        7.08e09,
        9.19e09,
        1.06e10,
        1.14e10,
        1.16e10,
    ]
)
intensity_bottom_predefined_profile_5 = np.array(
    [
        0,
        2.79e08,
        9.86e08,
        2.43e09,
        4.53e09,
        6.81e09,
        8.93e09,
        1.04e10,
        1.12e10,
        1.15e10,
        1.16e10,
    ]
)


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n OVIII sum_two_gauss_profiles"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(slits_number, intensities_top_gauss_profiles, label="top_to_bottom opening")
plt.plot(slits_number, intensities_bottom_gauss_profiles, label="bottom_to_top opening")
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n OVIII predefined_profile"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(
    slits_number, intensity_top_predefined_profile_5, label="top_to_bottom opening"
)
plt.plot(
    slits_number, intensity_bottom_predefined_profile_5, label="bottom_to_top opening"
)
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


### zapisac procentowa zmiane sygnalu - niezaleznie od profilu - mozna to wtedy ladnie wyliczyc

print("\n\nOVIII relative transmission change: ")
for j, i in enumerate(intensities_top_gauss_profiles):
    print(j + 1, "slit:", int(i / intensities_top_gauss_profiles[-1] * 100), "%")


"""N"""

intensities_top_gauss_profiles = np.array(
    [
        0,
        1.00e09,
        3.94e09,
        1.03e10,
        1.95e10,
        2.93e10,
        3.72e10,
        4.19e10,
        4.43e10,
        4.49e10,
        4.49e10,
    ]
)
intensities_bottom_gauss_profiles = np.array(
    [
        0,
        6.66e04,
        3.27e08,
        1.67e09,
        4.35e09,
        8.84e09,
        1.44e10,
        1.96e10,
        2.32e10,
        2.49e10,
        2.54e10,
    ]
)

# intensity_top_predefined_profile_5 = np.array([])
# intensity_bottom_predefined_profile_5 = np.array([])


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n N VII sum_two_gauss_profiles"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(slits_number, intensities_top_gauss_profiles, label="top_to_bottom opening")
plt.plot(slits_number, intensities_bottom_gauss_profiles, label="bottom_to_top opening")
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


# plt.title("Dependence ot the incoming light intensity on the number of slits\n N VII predefined_profile")
# plt.ylabel('Emissivity [ph/cm3/s]')
# plt.xlabel('Nr of slits')
# plt.plot(slits_number, intensity_top_predefined_profile_5, label="top_to_bottom opening")
# plt.plot(slits_number, intensity_bottom_predefined_profile_5, label="bottom_to_top opening")
# plt.legend(loc="upper left")
# for i in slits_number:
#     plt.axvline(i, 0, 1, label='pyplot vertical line',ls='--', color = "black", lw = 0.5)
# plt.show()


### zapisac procentowa zmiane sygnalu - niezaleznie od profilu - mozna to wtedy ladnie wyliczyc

print("\n\nN VII relative transmission change: ")
for j, i in enumerate(intensities_top_gauss_profiles):
    print(j + 1, "slit:", int(i / intensities_top_gauss_profiles[-1] * 100), "%")


"""B"""

intensities_top_gauss_profiles = np.array(
    [
        0,
        7.51e08,
        2.56e09,
        6.02e09,
        1.06e10,
        1.65e10,
        2.26e10,
        2.85e10,
        3.32e10,
        3.63e10,
        3.81e10,
    ]
)
intensities_bottom_gauss_profiles = np.array(
    [
        0,
        1.14e09,
        3.04e09,
        5.89e09,
        9.49e09,
        1.32e10,
        1.68e10,
        1.96e10,
        2.17e10,
        2.28e10,
        2.33e10,
    ]
)

# intensity_top_predefined_profile_5 = np.array([])
# intensity_bottom_predefined_profile_5 = np.array([])


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n B V sum_two_gauss_profiles"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(slits_number, intensities_top_gauss_profiles, label="top_to_bottom opening")
plt.plot(slits_number, intensities_bottom_gauss_profiles, label="bottom_to_top opening")
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


plt.title(
    "Dependence ot the incoming light intensity on the number of slits\n N VII predefined_profile"
)
plt.ylabel("Emissivity [ph/cm3/s]")
plt.xlabel("Nr of slits")
plt.plot(
    slits_number, intensity_top_predefined_profile_5, label="top_to_bottom opening"
)
plt.plot(
    slits_number, intensity_bottom_predefined_profile_5, label="bottom_to_top opening"
)
plt.legend(loc="upper left")
for i in slits_number:
    plt.axvline(i, 0, 1, label="pyplot vertical line", ls="--", color="black", lw=0.5)
plt.show()


### zapisac procentowa zmiane sygnalu - niezaleznie od profilu - mozna to wtedy ladnie wyliczyc

print("\n\nB VI relative transmission change: ")
for j, i in enumerate(intensities_top_gauss_profiles):
    print(j + 1, "slit:", int(i / intensities_top_gauss_profiles[-1] * 100), "%")
