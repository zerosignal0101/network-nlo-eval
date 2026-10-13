# A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems

**Abstract**—We investigate the impact of stimulated Raman scat-
tering (SRS) in multi - band transmissions. We first derive and test
against numerical results an extended Christodoulides - Zirngibl
(ECZ) SRS model based on a triangular approximation of the
Raman gain. Then, we exploit the ECZ SRS model in the context of
the Gaussian noise (GN) model. We generalize some known closed -
form expressions of the self - and cross - phase modulation variances,
as well as the amplified spontaneous emission noise variance, over
multi - band systems of bandwidth wider than the SRS bandwidth,
i.e., beyond the C+L band. Finally, the proposed theoretical model is
used to estimate the performance of multi - band systems in terms of
signal - to - noise ratio (SNR) and achievable information rate (AIR).

**Index Terms**—Stimulated Raman scattering (SRS), Gaussian
noise (GN) model, multi - band, ultra - wide band.

## I. INTRODUCTION

MULTI - BAND transmissions represent a viable strategy to
enhance the throughput of optical systems by increasing
the signal bandwidth [1]. The main advantage of wavelength
division multiplexing (WDM) multi - band systems compared to
other high - capacity system designs lies in the capability to be
used over single - mode fibers (SMFs), which already populate
the optical networks.

However, while the technology underpinning SMF transmis -
sions is well consolidated for the so - called C band (1530–
1565 nm), the transmission over wider bandwidths must ad -
dress several technological challenges. For instance, multi - band
amplification techniques are needed to support such systems.
While common erbium - doped fiber amplifiers (EDFAs) canstill be used to amplify the L band (1565 - 1625 nm) at the expense of a higher noise figure, the amplification of the S (1460 - 1530 nm) and the E (1360 - 1460 nm) bands requires different rare - earth elements, such as Thulium and Neodymium [2]. Other possible strategies are represented by distributed Raman amplifiers (DRA) [3], [4] and semiconductor optical amplifiers (SOAs) [5], which inherently offer wide amplification bandwidths.

However, the response of the optical fiber to such wide bandwidths becomes non - trivial. First of all, the fiber parameters such as attenuation and dispersion have a non - negligible dependency on frequency, such that WDM channels belonging to different bands undergo linear effects with different strengths [6]. While the linear effects can be compensated, their frequency dependence yields complex nonlinear interactions among spectral components through the fiber Kerr effect. The different bands interact nonlinearly also due to stimulated Raman scattering (SRS), through which low frequencies are amplified at the expense of higher - frequency components [6].

The design of multi - band systems accounting for the impact of all such aspects on the quality of transmission (QoT) represents a challenging task. An accurate QoT estimation can be achieved by numerically simulating the optical system by solving the optical field propagation along the fiber through split - step Fourier method (SSFM) simulations. However, the computational effort and time required by such a method escalate very quickly with the signal bandwidth [7].

In this context, semi - analytical perturbative models offer a significant complexity reduction. Of particular interest is the family of models that treats the nonlinear impairments as additive nonlinear interference (NLI). Among these models, the Gaussian noise (GN) model [8] stands out for its simplicity providing an estimate of the NLI power in the presence of Gaussian distributed symbols. The accuracy of the estimation is further improved by accounting for the dependency of the NLI on the modulation format of the transmitted symbols, as done by the enhanced GN model [9]. Ultra - wideband extensions of the GN model to include SRS were proposed in [10], [11], [12], and later exploited to analyze multi - band optical systems in [1], [13], [14], [15]. In particular, the work in [12] exploited the Christodoulides - Zirngibl formula of the signal power evolution along distance in the presence of SRS [16], [17], which was derived under the assumption of linear Raman gain vs frequency.

Along the same lines, the modulation format was included in [18].

The search for closed - form expressions for the SRS - aware NLI power has gained interest since it can bring the ultimate complexity reduction by allowing real - time estimations. Notably, the work in [19] proposed closed - form expressions of the semi - analytical estimate of the variance of self - phase modulation (SPM) and cross - phase modulation (XPM) of [12], based on [16], [17]. However, the linear approximation of the Raman gain of [16], [17] is justified when the WDM bandwidth is below $\approx15$ THz, i.e., up to the C + L band, while it over - estimates the Raman gain for larger bandwidths. To overcome this limitation, the works in [20], [21], [22], [23] fitted the numerical solution of the power evolution along distance to extend the GN closed - form formulas.

In this work, which is an extended version of [24], we propose to leverage a truncated triangular approximation of the Raman gain to extend the Christodoulides - Zirngibl (CZ) SRS formula for the signal power evolution derived in [16], [17]. We show that the proposed extended CZ (ECZ) SRS model can be used to extend the GN model formulas in [19] to multi - band transmissions even beyond the C + L band and we compare it against a numerical implementation of the GN model. Moreover, the SRS formula can be used to estimate even the SRS - inflated amplified spontaneous emission (ASE) power, thus yielding an approximated real - time estimation of the signal - to - noise ratio (SNR).

The remainder of this paper is organized as follows. In Section II, we tackle the topic of signal power evolution and derive the ECZ SRS model based on the truncated triangular Raman gain approximation. In Section III we generalize the SPM and XPM variance formulas based on the novel ECZ SRS model. Section IV focuses on the estimation of QoT metrics. Finally, in Section V we draw our conclusions.

## II. POWER EVOLUTION WITH SRS

The evolution along the fiber - optic propagation distance $z$ of the $i$th WDM channel power $P_i$ is governed by the following ordinary differential equation (ODE) [6], for $i = 1,\ldots,N_{\text{ch}}$:

$$
\begin{align*}
\frac{\mathrm{d}P_i}{\mathrm{d}z}&=-P_i\sum_{j = 1}^{i - 1}\frac{\lambda_i}{\lambda_j}\tilde{g}_R(\vert f_i - f_j\vert)P_j\\
&+P_i\sum_{j = i + 1}^{N_{\text{ch}}}\tilde{g}_R(\vert f_i - f_j\vert)P_j-\alpha(f_i)P_i
\end{align*}
$$

where $\lambda_i$ is the carrier wavelength of the $i$th channel, $f_i$ is its low - pass frequency in a reference system centered at the WDM central frequency, $N_{\text{ch}}$ is the number of channels, $\alpha$ is the fiber loss coefficient, and $\tilde{g}_R$ the polarization - averaged Raman gain coefficient. The first term on the right - hand side of (1) accounts for the depletion of channel $i$ to amplify lower frequency channels, while the second term accounts for the power acquired from higher frequency channels.

Fig. 1(a) shows the fiber loss coefficient, as a function of the wavelength $\lambda$, obtained through a quadratic polynomial fitting, in a dB scale, of the experimental attenuation profile available 

in [20]:

$$\alpha [\text{dB}/\text{km}] \approx \alpha_2(\lambda - \lambda_0)^2 + \alpha_1(\lambda - \lambda_0) + \alpha_0$$

with $\lambda_0 = 1550$ nm and coefficients: $\alpha_2 = 3.7685\times10^{-6}[\text{dB}/(\text{km}\cdot\text{nm}^2)], \alpha_1=-7.3764\times 10^{-5}[\text{dB}/(\text{km}\cdot\text{nm})]$, and $\alpha_0 = 0.162[\text{dB}/\text{km}]$. We observe a negligible variation over the C or the C+L band since the maximum excursion is 0.016 dB/km. On the other hand, such an excursion becomes significant for wider bandwidths, reaching 0.15 dB/km over the E+S+C+L bands. Fig. 1(b) shows the Raman gain coefficient $\tilde{g}_R(|f_i - f_j|)$, interpolated from [6], as a function of the frequency shift $\Delta f = f_i - f_j$ between channel $i$ and $j$.

Since the power evolution in the general form of (1) does not have an analytical solution, it needs to be solved numerically for each WDM channel. We next review the approximated analytical solution of (1) proposed in [16], [17], which is valid for frequency shifts within $\approx 15$ THz, and then we propose a generalization for wider bandwidths.

### A. The Christodoulides-Zirngibl SRS Model

An analytical solution of (1) has been derived in [16], [17] under the following assumptions: i) a linear dependency of $\tilde{g}_R(\Delta f)$ on the frequency shift, namely $\tilde{g}_R(\Delta f) \approx C_{\text{r}}\Delta f$, ii) a frequency - independent fiber attenuation coefficient $\bar{\alpha}$, and iii) $\frac{\lambda_i}{\lambda_j} \approx 1$. The power evolution equation then simplifies as follows:

$$
\frac{\text{d}P_i}{\text{d}z} = P_i\sum_{j = 1}^{N_{\text{ch}}}C_{\text{r}}(f_j - f_i)P_j - \bar{\alpha}P_i\tag{2}
$$

where the first term in the right - hand side accounts both for am - plification and depletion, depending on the sign of $f_j - f_i$. The coefficient $C_{\mathrm{r}}$ represents the slope of the linear approximation of the Raman gain coefficient, as sketched in Fig. 1, and can be found through a linear regression on the experimental data. Along the lines of [17], the system of coupled ODEs in (2) can be written as a single equation in terms of the WDM power spectral density (PSD) $\mathcal{G}(z,f)$ instead of channel powers, namely:

$$
\frac{\mathrm{d}\mathcal{G}(z,f)}{\mathrm{d}z}=\mathcal{G}(z,f)C_{\mathrm{r}}\int_{-\infty}^{\infty}(\mu - f)\mathcal{G}(z,\mu)\mathrm{d}\mu-\bar{\alpha}\mathcal{G}(z,f)
$$
(3)

where the summation over the channels was replaced by inte - gration in frequency. A key property of (3) is that the total WDM power at any coordinate $z$, defined as $P_{\mathrm{t}}(z)\triangleq\int_{-\infty}^{\infty}\mathcal{G}(z,\mu)\mathrm{d}\mu$, is set by the fiber loss only. In fact, by integrating both sides of (3) with respect to frequency, the first term on the right - hand side disappears resulting in $P_{\mathrm{t}}(z)=P_{\mathrm{t}}e^{-\bar{\alpha}z}$, where $P_{\mathrm{t}}\equiv P_{\mathrm{t}}(0)$ is the WDM total power injected in the fiber. Therefore, dividing (3) by $\mathcal{G}(z,f)$ and taking the derivative with respect to frequency yields:

$$
\frac{\mathrm{d}}{\mathrm{d}f}\frac{\mathrm{d}\ln\mathcal{G}(z,f)}{\mathrm{d}z}=-C_{\mathrm{r}}P_{\mathrm{t}}e^{-\bar{\alpha}z}
$$
(4)

Following [17], (4) can be integrated first with respect to distance and then to frequency, thus obtaining:

$$
\ln\left(\frac{\mathcal{G}(z,f)}{\mathcal{G}(0,f)}\right)=-C_{\mathrm{r}}P_{\mathrm{t}}L_{\mathrm{eff}}(z)f + F(z)
$$
(5)

where $L_{\mathrm{eff}}(z)=(1 - e^{-\bar{\alpha}z})/\bar{\alpha}$ is the fiber effective length at coordinate $z$, and $F(z)$ an integration constant. By imposing the conservation of the total power we get:

$$
\mathcal{G}(z,f)=\mathcal{G}(0,f)\frac{e^{-\bar{\alpha}z}e^{-C_{\mathrm{r}}P_{\mathrm{t}}L_{\mathrm{eff}}(z)f}P_{\mathrm{t}}}{\int_{-\infty}^{\infty}\mathcal{G}(0,\mu)e^{-C_{\mathrm{r}}P_{\mathrm{t}}L_{\mathrm{eff}}(z)\mu}\mathrm{d}\mu}
$$
(6)

From (6) it can be seen that the SRS induces a frequency - dependent tilt in the signal power profile without affecting the total power, which decreases only through the fiber loss.

### B. The Extended CZ SRS Model

Fig. 1(b) suggests that a linear approximation of the Raman gain is justified when the maximum frequency shift is less than $\approx15$ THz, as the Raman gain coefficient quickly vanishes at wider spacings. For larger bandwidths, the Raman gain co - efficient is better described through the so - called triangular approximation [25], where the gain is treated as a linear function

up to a cut - off shift $\Delta f_{\mathrm{co}}$ and forced to zero at higher frequencies, as sketched in Fig. 1(b). Under this approximation, a generic frequency $f$ interacts with the neighboring ones only within an effective window, experiencing amplification/depletion from the higher/smaller frequency channels, respectively. We define the effective window as the intersection between the WDM bandwidth and the two - sided Raman triangular window of bandwidth $2\Delta f_{\mathrm{co}}$. The idea is sketched in Fig. 2, where $\nu_1$ and $\nu_2$ represent the edges of such an effective window, while $f_{\mathrm{m}}$ and $f_{\mathrm{M}}$ are the minimum and maximum frequencies in the WDM bandwidth, respectively. It is worth noting that the special case ① coincides with the scenario analyzed in the previous section since the whole WDM bandwidth $B_{\mathrm{t}}$ is covered by the Raman window.

In this more general context, (3) can be generalized to:

$$
\frac{\mathrm{d}\mathcal{G}(z,f)}{\mathrm{d}z}=\mathcal{G}(z,f)C_{\mathrm{r}}\int_{\nu_1(f)}^{\nu_2(f)}(\mu - f)\mathcal{G}(z,\mu)\mathrm{d}\mu-\overline{\alpha}\mathcal{G}(z,f)
$$
(7)

where the frequency integral now spans only the frequencies within the effective window, whose limits are:

$$
\nu_2(f)=\min(f + \Delta f_{\mathrm{co}},f_{\mathrm{M}})
$$
(8)

$$
\nu_1(f)=\max(f_{\mathrm{m}},f - \Delta f_{\mathrm{co}})
$$
(9)

The difference with respect to the case analyzed in the previous section is thus the frequency dependency of the integral limits in (7). We thus exploit the Leibniz integral rule obtaining:

$$
\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}f}\frac{\mathrm{d}\ln\mathcal{G}(z,f)}{\mathrm{d}z}&=C_{\mathrm{r}}\left(\mathcal{G}(z,\nu_2(f))(\nu_2(f)-f)\frac{\mathrm{d}\nu_2(f)}{\mathrm{d}f}\right.\\
&-\mathcal{G}(z,\nu_1(f))(\nu_1(f)-f)\frac{\mathrm{d}\nu_1(f)}{\mathrm{d}f}-P_{\mathrm{I}}(z;\nu_1(f),\nu_2(f))\Bigg)
\end{align*}
$$
(10)

where we introduced the definition of integrated power in the effective window:

$$
P_{\mathrm{I}}(z;\nu_1\nu_2)\triangleq\int_{\nu_1}^{\nu_2}\mathcal{G}(z,\mu)\mathrm{d}\mu
$$
(11)

which coincides with the total power at coordinate $z$ only when the Raman window is wider than the WDM bandwidth. Since $\mathcal{G}(z,\nu_{1,2})\neq0$ only for $f_{\mathrm{m}}\leq\nu_{1,2}\leq f_{\mathrm{M}}$, we can remove the min/max of (8)-(9) by $\nu_{1,2}\equiv f\pm\Delta f_{\mathrm{co}}$. This allows us to express (10) in the compact form:

$$
\frac{\mathrm{d}}{\mathrm{d}f}\frac{\mathrm{d}\ln\mathcal{G}(z,f)}{\mathrm{d}z}=C_{\mathrm{r}}(P_{\mathrm{I},\mathrm{T}}(z;f,\Delta f_{\mathrm{co}})-P_{\mathrm{I}}(z;\nu_1(f),\nu_2(f)))
$$
(12)

where we introduced the following definition:

$$P_{\mathrm{I}, \mathrm{T}}(z ; f, \Delta f_{\mathrm{co}}) \triangleq\left(\mathcal{G}(z, f-\Delta f_{\mathrm{co}})+\mathcal{G}(z, f+\Delta f_{\mathrm{co}})\right) \Delta f_{\mathrm{co}}. \quad(13)$$

It is worth noting that $P_{\mathrm{I}, \mathrm{T}}(z ; f, \Delta f_{\mathrm{co}})$ coincides with the PSD integrated in the $2 \Delta f_{\mathrm{co}}$ Raman window through the trapezoidal rule. Fig. 2 shows the relation between $P_{\mathrm{I}}$ and $P_{\mathrm{I}, \mathrm{T}}$ in the four cases under investigation. In particular, the two quantities are equal when the Raman window is narrower than the WDM band - width. Equation (12) thus generalizes (4), and coincides with it in the special case ① depicted in Fig. 2, where the integrated power coincides with the total power and $P_{\mathrm{I}, \mathrm{T}}(z ; f, \Delta f_{\mathrm{co}})=0$.

Equation (12) does not have an analytical solution. We thus proceed by introducing the following key approximations:

$$
P_{\mathrm{I}}(z ; \nu_{1}(f), \nu_{2}(f)) \approx P_{\mathrm{I}}(0 ; \nu_{1}(f), \nu_{2}(f)) e^{-\bar{\alpha} z} \quad(14)
$$

$$
P_{\mathrm{I}, \mathrm{T}}(z ; f, \Delta f_{\mathrm{co}}) \approx P_{\mathrm{I}, \mathrm{T}}(0 ; f, \Delta f_{\mathrm{co}}) e^{-\bar{\alpha} z} \quad(15)
$$

which assume that the power is undepleted by SRS within a window (effective or Raman, respectively). Such an assumption holds when the window coincides with the WDM bandwidth and the attenuation coefficient is constant with respect to frequency, as in Section II - A.

Along the lines of Section II - A, we then integrate with respect to distance and frequency, obtaining:

$$
\ln \left(\frac{\mathcal{G}(z, f)}{\mathcal{G}(0, f)}\right)=-C_{\mathrm{r}} L_{\mathrm{eff}}(z) r(f)+F(z) \quad(16)
$$

which generalizes (5) through the Raman shaping profile $r(f)$. Such a term results from the evaluation of the following indefinite integral:

$$
r(f)+F(z)=\int P_{\mathrm{I}}(0 ; \nu_{1}(f), \nu_{2}(f)) \mathrm{d} f-\int P_{\mathrm{I}, \mathrm{T}}(0 ; f, \Delta f_{\mathrm{co}}) \mathrm{d} f \quad(17)
$$

With uniform launched PSD $\mathcal{G}(0, f)=P_{\mathrm{t}} / B_{\mathrm{t}}$, we show in the Appendix that the Raman shaping profile can be expressed as:

$$
r(f)=\begin{cases}\frac{P_{\mathrm{t}} f}{} & \text { case } ① \\ 0 & \text { case } ② \\ \frac{P_{\mathrm{t}}}{B_{\mathrm{t}}}\left(\frac{f^{2}}{2}-f f_{\mathrm{m}}+\frac{f_{\mathrm{M}}^{2}-\Delta f_{\mathrm{co}}^{2}}{2}\right) & \text { case } ③ \\ \frac{P_{\mathrm{t}}}{B_{\mathrm{t}}}\left(f_{\mathrm{M}} f-\frac{f^{2}}{2}-\frac{f_{\mathrm{m}}^{2}-\Delta f_{\mathrm{co}}^{2}}{2}\right) & \text { case } ④\end{cases} \quad(18)
$$

for the four cases sketched in Fig. 2. In the simplest scenario, case ①, the Raman window encompasses the whole WDM bandwidth $B_{\mathrm{t}}$. Therefore, $P_{\mathrm{I}} \equiv P_{\mathrm{t}}$ and $P_{\mathrm{I}, \mathrm{T}} = 0$ yielding $r(f)=P_{\mathrm{t}} f$ as in the original theory [16], [17]. On the other hand, if the Raman window covers only an inner portion of the WDM bandwidth, $r(f)$ balances out to zero (case ②) since $P_{\mathrm{I}}=P_{\mathrm{I}, \mathrm{T}}$. For all the other scenarios, the term $r(f)$ takes one of the expressions labeled with ③ or ④ in (18), depending on the position of the frequency of interest within the WDM.

The integration constant $F(z)$ is found by imposing the conservation of the total power, similarly to the previous section, and thus yielding the following solution:

$$
\mathcal{G}(z, f) \simeq \mathcal{G}(0, f) \frac{e^{-C_{\mathrm{r}} L_{\mathrm{eff}}(z) r(f)-\bar{\alpha} z} P_{\mathrm{t}}}{\int_{-\infty}^{\infty} \mathcal{G}(0, \mu) e^{-C_{\mathrm{r}} L_{\mathrm{eff}}(z) r(\mu)} \mathrm{d} \mu} \quad(19)
$$

which is formally identical to (6) with the replacement of $P_{\mathrm{t}}f \to r(f)$ in the exponential terms. Since $r(f)$ is slowly varying over a channel bandwidth, we can substitute $\mathcal{G}$ with a train of Dirac’s delta obtaining the following equation for $P_{i}$:

$$
P_{i}(z)\simeq P\frac{e^{-C_{\mathrm{r}}L_{\mathrm{eff}}(z)r(f_{i})-\alpha_{i}z}P_{\mathrm{t}}}{\sum_{j = 1}^{N_{\mathrm{ch}}}Pe^{-C_{\mathrm{r}}L_{\mathrm{eff}}(z)r(f_{j})}}\qquad(20)
$$

where $P$ is the launch power per channel and we also arbitrarily chose as the common fiber attenuation $\bar{\alpha}$ within a window its value at the channel under test, i.e., $\alpha_{i}=\alpha(f_{i})$ with $L_{\mathrm{eff}}(z)=(1 - e^{-\alpha_{i}z})/\alpha_{i}$, as this choice provides the best fit with simulations.

### C. Numerical Results

We test the proposed analytical solution in (20) against a numerical solution of the ODEs in (1), which is based on the true attenuation profile and Raman gain coefficient in Fig. 1, through the MATLAB function ode45. We chose the slope $C_{\mathrm{r}}$ and the cutoff frequency shift $\Delta f_{\mathrm{co}}$ by minimizing the least - squares error over the WDM bandwidth between the solution of (1) with the true $\tilde{g}_{\mathrm{R}}$ and the triangular approximation.

In a first test, we considered the transmission of 64 Gbaud channels spaced 75 GHz, covering either the C+L (11.4 THz), or the S+C+L (20.8 THz), or the E+S+C+L (35.9 THz) band without any guard band intervals. The total power was set to 21 dBm, while the link consisted of $L = 100$ km of an SMF. For the generic $i$th WDM channel, we define the span SRS gain as the ratio between the channel power at fiber - end and its value without the SRS, namely:

$$
\mathrm{SRS}\ \mathrm{gain}(f_{i})\triangleq\frac{P_{i}(L)}{P_{i,\mathrm{noSRS}}(L)}.\qquad(21)
$$

Fig. 3 shows the SRS gain computed both with the numerical solution of the ODE (markers) and by using the proposed formula (solid line) in (20). For completeness, we show with a dashed line the CZ SRS solution [16], [17] in (6). For the sake of comparison, in (6) we chose the constant attenuation coefficient as in (20). The figure shows a good agreement between the numerical results and the proposed ECZ model, with a root mean square error (RMSE) of 0.07 dB, 0.18 dB, and 0.29 dB, for the C+L, S+C+L, and E+S+C+L transmission, respectively. For the same transmissions, the CZ model (linear approximation of the Raman gain) yields an RMSE of 0.07 dB, 0.6 dB, and 2.9 dB.

From the figure, we note that over the C+L band the triangular and the linear SRS approximations are overlapping. This is not surprising, since the Raman window encompasses the entire WDM band. Differences start appearing in the S+C+L case, mainly in the S and L band, where there are edge channels that fall in the cases ③ and ④, since their Raman window does not cover the whole bandwidth. It can be seen from Fig. 3, that the SRS impact on these regions is smaller than the one predicted by [16], [17] since only a subset of the overall channels participates in it. Thanks to the triangular shape of the Raman gain adopted in the proposed model, (20) is able to capture such a phenomenon. On the other hand, in the central WDM region, the Raman window encompasses the whole WDM bandwidth (case ① of Fig. 2) and thus the SRS gain is well predicted by the CZ formula. In the E+S+C+L band, the linear SRS approximation overestimates the Raman effect all over the WDM bandwidth by artificially increasing the gain tilt. It is worth noting that in such a case an inner portion of channels meets the conditions of case ② in Fig. 2, which, in the proposed theory, have zero Raman shaping profile in (18). The numerical results reported in Fig. 3 are in agreement with such a prediction, exhibiting an SRS gain $\approx$ 0 dB in the central part of the bandwidth.

A key assumption of the proposed model is represented by the undepleted integrated power in (14)–(15). This assumption becomes critical in the presence of strong SRS, for instance when the total WDM power $P_t$ is high. We thus investigated selected cases of $P_t$ focusing on the S+C+L transmission case. The results are shown in Fig. 4 for $P_t = 21$ dBm, 23 dBm, and 25 dBm corresponding to a channel power of $-3.4$ dBm, $-1.4$ dBm, and 1.4 dBm, respectively. We note that the ECZ SRS model is able to capture the qualitative three - region shape of the SRS gain. Nevertheless, the accuracy with respect to the Runge - Kutta results decreases in the edge regions as the power increases, with an RMSE of 0.18 dB, 0.3 dB, and 0.8 dB, respectively.

We then investigated the simple yet relevant case of signal power pre - emphasis in the form of an opposite - sign SRS gain. Namely, this power allocation aims to pre - compensate a portion of the span SRS gain by applying a pre - emphasis to an uniform launched power. As an approximation, we still use (20) with:

$$
C_{\mathrm{r}}L_{\mathrm{eff}}(L)r(f)\to(1 - \bar{k})C_{\mathrm{r}}L_{\mathrm{eff}}(L)r(f)\tag{22}
$$

with $\bar{k}$ the pre - emphasis factor [18].

Fig. 5 shows the SRS gain for the S+C+L band transmission setup of Fig. 3 with the different power allocations reported in the inset. In particular, we fixed the total power to 21 dBm and then applied a pre - emphasis with a factor $\bar{k}\in(0,0.5,1)$. The numerical results (markers) show that the pre - emphasis effectively reduces the SRS gain experienced by the signal. We then compared the results with the formula in (20) combined with (22), which is reported with solid lines. The RMSE is within 0.2 dB in all cases, justifying the use of (20) even with a non - flat power allocation.

## III. CLOSED - FORM GN MODEL

In this Section, we exploit the novel ECZ SRS model in the context of the GN model. In this framework, the NLI at channel $f_{i}$ is treated as an additive noise [26] whose variance, after compensation of the linear effects and SRS on the signal, is [8], [19]:

$$
\sigma_{\mathrm{NLI}}^{2}(f_{i})=\frac{16}{27}B_{i}\iint_{-\infty}^{\infty}\gamma^{2}(f_{i},f_{1},f_{2})\left|\eta(f_{i},f_{1},f_{2})\right|^{2}\times\mathcal{G}(0,f_{i}+f_{1})\mathcal{G}(0,f_{i}+f_{1}+f_{2})\mathcal{G}(0,f_{i}+f_{2})\mathrm{d}f_{1}\mathrm{d}f_{2}\tag{23}
$$

with $B_{i}$ the channel bandwidth, $\gamma(f,f_{1},f_{2})$ the fiber nonlinear coefficient$^{1}$, and $\eta(f,f_{1},f_{2})$ the link kernel (or link function) weighting the four - wave mixing interaction among signal frequencies, which can be written as:

$$
\eta(f,f_{1},f_{2})=\sum_{k = 0}^{N - 1}e^{j\Delta\beta kL}\int_{0}^{L}e^{j\Delta\beta\zeta}\sqrt{\rho(\zeta,f,f_{1},f_{2})}\mathrm{d}\zeta\tag{24}
$$

for a dispersion uncompensated link composed of $N$ identical spans, each of length $L$, with lumped amplification in each span followed by a gain - flattening filter (GFF) or a dynamic gain equalizer (DGE) to restore the signal PSD. The link kernel accounts for the fiber loss and the SRS experienced by the signals along the fiber through the following function:

$$
\rho(\zeta, f, f_1, f_2)=\frac{\mathcal{R}(\zeta, f + f_1 + f_2)\mathcal{R}(\zeta, f + f_1)\mathcal{R}(\zeta, f + f_2)}{\mathcal{R}(\zeta, f)}(25)
$$

with $\mathcal{R}(z, f)\triangleq\mathcal{G}(z, f)/\mathcal{G}(0, f)$ the loss/gain profile at coordinate $z$ and frequency $f$. Equation (24) also includes the fiber chromatic dispersion through the phase matching coefficient $\Delta\beta\triangleq4\pi^{2}f_1f_2[\beta_2+\pi(2f + f_1 + f_2)\beta_3]$, with $\beta_2$ the dispersion and $\beta_3$ its slope. It is worth noting that the integral in (24) accounts for the nonlinearity arising locally within the span, while the outer summation keeps track of the dispersive effects accumulated in the previous spans.

The NLI variance in (23) can be evaluated by means of numerical integration in frequency and in distance. The integral over the distance can be approximated by following similar steps of [19]:

$$
\int_{0}^{L}e^{j\Delta\beta\zeta}\sqrt{\rho(\zeta, f, f_1, f_2)}\mathrm{d}\zeta\approx(1 - S)\eta_1(\bar{\alpha})+S\eta_1(2\bar{\alpha})\ (26)
$$

where $\eta_1(\bar{\alpha})$ is the single - span kernel without SRS [8]:

$$
\eta_1(\bar{\alpha})\triangleq\frac{1 - e^{-\bar{\alpha}L}e^{j\Delta\beta L}}{\bar{\alpha}-j\Delta\beta}\ (27)
$$

while the factor $S$ accounts for the presence of SRS through the Raman shaping profile $r(f)$ in (18) as follows:

$$
S\triangleq\frac{C_{\mathrm{r}}}{2\bar{\alpha}}(r(f + f_1 + f_2)+r(f + f_1)+r(f + f_2)-r(f)).\ (28)
$$

The approximation sign in (26) accounts for the first - order Taylor expansion $e^{-C_{\mathrm{r}}L_{\mathrm{eff}}(\zeta)r(f)}\approx1 - C_{\mathrm{r}}L_{\mathrm{eff}}(\zeta)r(f)$ within $\mathcal{R}(\zeta, f)$. For a homogeneous link, the external summation in (24) can be closed in the so - called phased - array term [8]. It is worth noting that (26) coincides with the solution in [19] in case ① of Fig. 2.

Since the power profile $\mathcal{R}(z, f)$ can be approximated as flat within a channel band, we can generalize the closed - form expressions of [19] by:

$$
\sigma_{\mathrm{SPM}}^{2}(f_i)\approx\frac{4}{9}\frac{P_{i}^{3}}{B_{i}^{2}}N^{1 + \varepsilon}\frac{\gamma_{i, i}^{2}\pi}{\phi_{i}3\alpha_{i}^{2}}e^{2C_{\mathrm{r}}\bar{k}L_{\mathrm{eff}}(L)r(f_i)}\times\left[\frac{T_i-\alpha_{i}^{2}}{\alpha_{i}}\mathrm{asinh}\left(\frac{\phi_{i}B_{i}^{2}}{\pi\alpha_{i}}\right)+\frac{4\alpha_{i}^{2}-T_i}{2\alpha_{i}}\mathrm{asinh}\left(\frac{\phi_{i}B_{i}^{2}}{2\pi\alpha_{i}}\right)\right]_{(29)}
$$

$$
\sigma_{\mathrm{XPM}}^{2}(f_i)\approx\frac{32}{27}P_iN\sum_{\ell = 1}^{N_{\mathrm{ch}}}\frac{P_{\ell}^{2}}{B_{\ell}}\frac{\gamma_{i, \ell}^{2}}{\phi_{i, \ell}3\alpha_{\ell}^{2}}e^{2C_{\mathrm{r}}\bar{k}L_{\mathrm{eff}}(L)r(f_{\ell})}\times\left[\frac{T_{\ell}-\alpha_{\ell}^{2}}{\alpha_{\ell}}\mathrm{atan}\left(\frac{\phi_{i, \ell}B_{i}}{\alpha_{\ell}}\right)+\frac{4\alpha_{\ell}^{2}-T_{\ell}}{2\alpha_{\ell}}\mathrm{atan}\left(\frac{\phi_{i, \ell}B_{i}}{2\alpha_{\ell}}\right)\right]_{(30)}
$$

with $\phi_i \triangleq \frac{3}{2}\pi^2(\beta_2 + 2\pi f_i\beta_3)$ and $\phi_{i,\ell} \triangleq 2\pi^2(f_{\ell} - f_i)[\beta_2 + \pi\beta_3(f_{\ell} + f_i)]$. The novelty of (29)–(30) relies in the SRS - dependent term $T_i$, which is here defined as:
$$T_i \triangleq (2\alpha_i - C_{\mathrm{r}}r(f_i))^2\tag{31}$$
and generalizes the term $(2\alpha_i - C_{\mathrm{r}}P_{\mathrm{t}}f_i)^2$ in [19]. Moreover, in (29)–(30) we included a pre - emphasis factor $\bar{k}$ by exploiting (20), (22) and the wavelength dependency of the fiber nonlinear coefficient [27]:

$$
\gamma_{i,\ell}=\frac{2\pi}{\lambda_i}\frac{2n_2}{A_{\mathrm{eff}}(\lambda_i)+A_{\mathrm{eff}}(\lambda_{\ell})}\tag{32}
$$

with $A_{\mathrm{eff}}(\lambda_i)$ the fiber effective area at the wavelength $\lambda_i$ and $n_2$ the fiber nonlinear index. The channel index $\ell$ discriminates between the nonlinear coefficient relative to SPM ($\ell = i$) or XPM ($\ell\neq i$). While the XPM variance accumulates incoherently with the number of spans, the factor $\varepsilon$ in (29) accounts for a coherent accumulation [8], [19], [28] of the SPM variance along the distance.

### A. Numerical Results

We tested the generalized closed - form expressions of the SPM and XPM variance in (29)–(30) against a fully - numerical implementation of the GN model. Such a benchmark is obtained by using the numerical solution of the ODE in (1) with the true profiles of Fig. 1 to obtain the function $\rho(\zeta,f,f_1,f_2)$ for the link kernel in (24). The NLI variance in (23) is then evaluated numerically by exploiting Filon’s method [18] for the integral along the distance in (24) and the Monte Carlo method [29] for the double frequency integral. The link under test was composed of $10\times100$ km of single - mode fibers, with full equalization of the SRS gain at the end of each span. The fibers had SPM non - linear coefficient and chromatic dispersion coefficient reported in Fig. 6. It is worth noting that the nonlinear coefficient has an opposite trend with respect to wavelength compared to the chromatic dispersion. Comparing Fig. 6 with the results reported in Figs. 3–5 we also note that the nonlinear coefficient trend is opposite to the one of the SRS gain. Throughout this section, we consider the transmission of a dual polarization WDM multi - band signal, where each channel was modulated with complex Gaussian distributed symbols. The channel symbol rate was 64 Gbaud and the spacing 75 GHz, with guard bands of 500 GHz between neighboring multi - bands.

We first tested the accuracy of the SPM and XPM variance closed - form expressions with respect to the channel power. Fig. 7 shows such a variance for a flat WDM launch power over the E+S+C+L band, normalized to the cube of the channel power. The results are reported for four channels of interest placed in the middle of the E, S, C, and L bands, and plotted versus the channel power. The markers represent the benchmark while the solid lines are the formulas in (29)–(30). We can see from the figures that the variance does not scale with $P^{3}$, contrary to what happens in absence of SRS [8]. This is caused by the SRS gain which makes the link kernel in (24) dependent on the WDM total power. In particular, we note that the SPM and XPM variance of the CUT in the central E band scales slower than $P^{3}$, while the scaling is faster in the L band. Moreover, the inset of Fig. 7 shows that, due to SRS, the balance between the two nonlinear effects depends both on the channel power and the band. The worst RMSE between the XPM variance obtained with the proposed theory and the numerical results is 0.5 dB for the CUT in the S - band, while in the other bands we obtained an RMSE of 0.2 dB. On the other hand, the estimation of SPM is more critical, with a worst - case RMSE of 1.3 dB in the L band. However, since XPM is the dominant impairment in the transmission under test, combining the two nonlinear effects yields an RMSE of 0.5 dB for the CUT in the S - band and 0.3 dB for the others.

Then, we considered the transmission of signals spanning the C+L, S+C+L, and E+S+C+L bands with fixed channel power equal to - 1 dBm. The results are shown in Fig. 8, where the markers indicate the benchmark GN model, and the solid lines are the generalized closed - form expressions in (29)–(30), while the dashed lines indicate the closed - form expressions derived in [19] for the C+L band assuming a linear Raman gain. For a fair comparison, in the latter expressions we used the same per - channel attenuation and wavelength - dependent fiber non - linear coefficient $\gamma$ as in (29)–(30) of this work. Fig. 8 (left) shows that in the C+L band the novel variance expressions coincide with those used in [19], and accurately estimate the NLI variance since the condition of case ① (see Fig. 2) is always met. We note that the NLI variance is almost flat with respect to wavelength, contrary to what was observed in [24]. This is due to the wavelength dependency of the nonlinear coefficient $\gamma$ reported in Fig. 6 (which was not accounted for in [24]) that partially counterbalances the SRS - induced tilt.

On the other hand, when the WDM signal covers the S+C+L bands as in Fig. 8 (center), the effect of the distributed interaction between the SRS and the Kerr effect is more visible, inducing a tilt on the NLI variance at the expense of the channels populating the L band. In this scenario, the generalized formulas of this work are in better agreement with the numerical estimation, with an RMSE of 0.4 dB against the 0.7 dB with a linear Raman gain coefficient (the dashed line). This result is consistent with comments drawn in Section II - C on the overestimation of the SRS gain in the WDM edges when a linear approximation is used. The linear approximation of the Raman gain becomes highly detrimental over wider bandwidths. In particular, for the E+S+C+L transmission in Fig. 8 (right), the proposed formulas allowed us to reduce the RMSE from 2 dB to 0.6 dB. It is worth noting that, in this last scenario, the NLI variance exhibits a tilted profile vs wavelength in the opposite direction compared to the one reported for the S+C+L band transmission. Such an NLI variance inflation in the E band is caused by the low values of the chromatic dispersion coefficient in this band [30], whose impact on the NLI out - weighs the one of SRS.

For the sake of completeness, in extra simulations, we computed the numerical GN model imposing the same triangular profile and attenuation value used in the theoretical formulas, and we obtained RMSE values comparable to those reported for Fig. 8. Such results suggest that the main source of model inaccuracy results from the assumption that the integrated power within a sub - band is undepleted by SRS, i.e., (14)-(15).

Finally, we tested the proposed formulas in the presence of a signal power pre - emphasis, hence with the extra approximation of (22). We focused on an S+C+L band transmission with a fixed total power of 23 dBm. We varied the amount of pre - compensated SRS gain by considering $\bar{k}=0,0.2$, and 0.5. All he other parameters were the same as Figs. 7 and 8. Fig. 9 shows the benchmark results through markers and the theoretical results with solid lines. The different power allocations are reported in the inset. Such a pre - emphasis is designed to pre - compensate all the accumulated SRS gain on the signal power when $\bar{k}=1$, which does not coincide with the SRS gain experienced by NLI variance since the NLI arises in a distributed way along propagation. In fact, the figure shows that the signal power pre - emphasis almost counterbalances the SRS - induced tilt on the NLI variance when $\bar{k}=0.2$ while it induces an opposite - sign tilt when $\bar{k}=0.5$. Although using the power profile derived in Section II - B with a non - uniform power allocation represents an approximation, the estimated RMSE between the NLI variance numerical results and the proposed formulas is within 0.5 dB in all three cases.

## IV. SNR ESTIMATION

By treating the ASE and the NLI as statistically independent additive noises, the SNR of a generic WDM channel, after SRS gain equalization on the signal power at the receiver, can be written as:

$$
\mathrm{SNR}(f_i) = \frac{P}{\sigma_{\mathrm{ASE}}^2(f_i) + \sigma_{\mathrm{NLI}}^2(f_i)} \tag{33} 
$$

where $\sigma_{\mathrm{NLI}}^2$ is the channel NLI variance and $\sigma_{\mathrm{ASE}}^2$ is the ASE noise variance. The variance of the ASE noise at the end of a link composed of $N$ identical spans can be generalized as:

$$
\sigma_{\mathrm{ASE}}^2(f_i) = hf_iF(f_i)B_iG(f_i)N \tag{34} 
$$

where $h$ is the Planck’s constant, $G(f_i)$ and $F(f_i)$ are the gain and noise figure of each line amplifier + DGE. The gain of the equalized amplifier restores the target signal PSD by equalizing the SRS and the fiber loss.

An ultra - fast theoretical estimation of the SNR in (33) can be obtained by exploiting the novel ECZ SRS model for the evaluation of both the ASE and the NLI variance. To test the accuracy of such an estimation, we compared the theoretical results with a fully - numerical SNR estimation relying on a Runge - Kutta solution of the ODEs in (1) for both the NLI and the ASE noise.

At first, we focused on an E+S+C+L transmission with total power equal to 25 dBm. All the system parameters were the same as in Section III. Each span had identical amplifiers with a per - band noise figure [30] equal to 6 dB (E band), 7 dB (S band), 5.5 dB (C band), and 6 dB (L band). Fig. 10 shows the SNR versus wavelength, with a good agreement between the theoretical SNR (line) and the exact SNR (markers). In the inset, we report the variances of the NLI and the ASE noise, showing that the E band is the most affected by both noise sources with particularly strong ASE noise, hence justifying the poor performance of such channels in terms of SNR.

To further explore the reliability of the proposed model in predicting the system SNR, we ran a campaign of simulations for many system setups. Namely, we randomly picked some of the link and WDM parameters, and we estimated the SNR for 1000 setup realizations. In particular, we picked the number of spans between 1 and 20, the chromatic dispersion between 8 and 18 ps/(nm·km), the symbol rate $R$ between 32 and 96 Gbaud, and the channel spacing between $1.2\times R$ and $1.7\times R$, the power $P$ level between - 4 and 0 dBm, and the pre - emphasis factor $\bar{k}$ between 0 and 1. All the parameters had uniform distributions. We collected 500 random seeds for an E+S+C+L band transmission and 500 for S+C+L band and evaluated the wavelength RMSE of the theoretical SNR. The analysis revealed that the most challenging scenarios for the model accuracy are those associated with high power. Fig. 11 shows the RMSE versus total power (left) and versus pre - emphasis (right). The figure shows an increasing trend of the RMSE with the total power, both in the S+C+L band (red circles) and in the E+S+C+L band (blue dots). We note that, for the E+S+C+L band setups, the high RMSE points observed in the left figure are associated with a small $\bar{k}$. This is because a small $\bar{k}$ increases the SRS gain tilt (see Fig. 5), such that some channels experience too high power over the E+S+C+L bands, thus outside the perturbative assumptions of the model. The pre - emphasis helps mitigate such a problem.

We next exploited the estimation of the SNR to evaluate the achievable information rate (AIR) for various transmissions. Treating both the signal and the NLI as Gaussian distributed, the AIR is given by the Shannon formula:

$$
\text{AIR}=2R\sum_{m = 1}^{N_{\text{ch}}}\log_2(1+\text{SNR}(f_m)).\quad(35)
$$

Despite the outer summation over the number of channels, the presence of fiber nonlinear effects makes the growth of the AIR with the WDM bandwidth slower than linear. In addition, the presence of SRS contributes to setting a strong dependency of the SNR on the frequency.

To investigate the scaling of the AIR with the transmitted bandwidth, we considered a gradual upgrade from the C band to the E+S+C+L band. For a fair comparison, in each case we adopted a power allocation (in the form of signal power pre - emphasis) maximizing the AIR. All the other parameters were identical to those used in Section II - C and Section III. Fig. 12 shows the spectral efficiency $\text{SE}=2\log_2(1+\text{SNR}(f))$ for the four transmissions under test and, in the inset, the corresponding AIR vs the WDM bandwidth. It can be seen that upgrading from C to C+L band, corresponding to a 65% increase in the bandwidth, yields a similar growth in terms of AIR (63%). Such a bandwidth upgrade penalizes the C band, which acts as a pump to the L band through SRS.

Moving towards the S - band, we note that while its channels boost the performance of the C band and, partially, the L band, they offer lower SE. Overall, the 45% bandwidth increase results in 39% of AIR increase. The same phenomenon is observed for the upgrade toward the E band, which offers a limited contribution of the E - band per se, however with positive implications on the other bands thanks to its SRS pumping effect, as shown in Fig. 12. In this case, increasing the bandwidth by 42% with respect to the S+C+L transmission yields an AIR that is 31% higher.

## V. CONCLUSION

Fast theoretical models for QoT estimation help alleviate the complex task of the design and the study of multi - band optical systems. In this work, we generalized the Christodoulides - Zirngibl SRS model of the signal power evolution along a fiber optic by leveraging a truncated triangular approximation of the Raman gain coefficient. The numerical results show that the proposed ECZ SRS formula is a viable tool to estimate in real - time the SRS - affected signal power, thus effectively mitigating the SRS overestimation of the CZ SRS formula in transmissions covering the S+C+L and E+S+C+L bands.

We showed that, by using the ECZ SRS formula, the GN model closed - form expressions of [19], originally derived with a linear Raman gain coefficient, based on the CZ model, can be extended to estimate the variance of SPM and XPM in the context of multi - band transmissions. We reported a significant reduction of the RMSE compared to the original GN model in [19] over the S+C+L and E+S+C+L bands.

The proposed analytical expression for the SRS gain can be used to quickly estimate even the SRS - affected ASE variance, and thus both the SNR and the AIR under the GN model assumptions. A random simulative campaign over 1000 different setups showed that, except for a few outliers, the proposed model estimates the SNR with an RMSE smaller than 0.5 dB up to a total power of 23 dBm over the E+S+C+L band. It is worth noting that using the GN model of [19] based on the linear approximation of the Raman gain, the RMSE diverges more rapidly with the total power, yielding an RMSE greater than 1 dB already with 19 dBm in the E+S+C+L band.

Finally, we exploited the analytical model to investigate the benefits in terms of AIR in upgrading the system bandwidth from the C to the E+S+C+L band. The analysis showed that the optical channels in the E band offer limited contributions to the AIR while boosting the performance on the S, C, and L bands through SRS, in agreement with the findings of [22].

## APPENDIX A

### EVALUATION OF THE RAMAN SHAPING PROFILE

In this section we report the steps adopted in the evaluation of the following expression:

$$
\int (P_{1}(0;\nu_{1}(f),\nu_{2}(f)) - P_{1,\mathrm{T}}(0;f,\Delta f_{\mathrm{co}}))\mathrm{d}f\quad(36)
$$

appearing in (17). Assuming that the transmitted PSD is flat, the integrated power can be expressed as:

$$
P_{\text{I}}(0; \nu_1(f), \nu_2(f)) = 
\begin{cases} 
\frac{P_{\text{t}}}{B_{\text{t}}} & \text{case } ① \\
2\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}} & \text{case } ②  \\
\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}}(f - f_{\text{m}} + \Delta f_{\text{co}}) & \text{case } ③ \\
\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}}(f_{\text{M}} - f + \Delta f_{\text{co}}) & \text{case } ④
\end{cases}
(37) 
$$

where the effective window edges $\nu_1(f)$ and $\nu_2(f)$, defined in (9) - (8), identify the four scenarios sketched in Fig. 2. Similarly, the integrated power in the Raman window with trapezoidal quadrature, defined in (13), can be written as:

$$
P_{\text{I,T}}(0; f, \Delta f_{\text{co}}) = 
\begin{cases} 
0 & \text{case } ① \\
2\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}} & \text{case } ② \\
\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}}H(f_{\text{M}} - f - \Delta f_{\text{co}}) & \text{case } ③ \\
\Delta f_{\text{co}}\frac{P_{\text{t}}}{B_{\text{t}}}H(f - \Delta f_{\text{co}} - f_{\text{m}}) & \text{case } ④
\end{cases}
(38) 
$$

where we used the Heaviside function $H(x)$ to set to zero the PSD outside the effective window. Note that in case $② P_{\text{I,T}} \equiv P_{\text{I}}$ in the case of flat PSD.

We then evaluate the indefinite integral of the integrated power expressions in (37)-(38) by exploiting the properties: i) $\int H(x)dx = xH(x)$, and ii) $\int H(-x)dx = x(1 - H(x))$. Finally, gluing the solutions at the domain edges we find the Raman shaping factor $r(f)$ in (18). For instance, the solution of case ③ has to coincide with ① when the effective window upper edge $\nu_2(f)=f + \Delta f_{\text{co}}$ reaches the WDM spectrum highest frequency $f_{\text{M}}$, i.e., $f = f_{\text{M}} - \Delta f_{\text{co}}$. 
