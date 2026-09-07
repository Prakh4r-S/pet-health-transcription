# Recording scripts

Six consultations to read aloud and record. Each has a matching case file
in `cases/` with the expected extraction already written and an empty
`transcript` field for you to fill in from the Whisper output.

## How to use these

1. Read the script aloud and record it. Two people is much better than
   one — Whisper's segmentation behaves differently on a genuine
   two-speaker recording, and a single voice reading both parts produces
   unnaturally clean turn-taking.
2. Record through the app itself if you can, so the audio goes through
   the same MediaRecorder mixing, the same webm encoding, and the same
   model as production. A pristine studio recording tests something you
   will never actually run.
3. Run the audio through the service and copy the transcript it produces
   into the `transcript` field of the matching case file. **Do not clean
   it up.** The garbled words are the point.
4. Leave the expected extraction as written — it was derived from the
   script, not the transcript. That is what makes this an end-to-end
   measurement rather than a test of the extraction step alone.

Read at a natural pace. Talk over each other slightly, hesitate, correct
yourself. A stilted read produces a transcript that is easier than
anything real.

---

## 13 — Weight-based dosing

Tests dosing expressed per kilogram rather than as an absolute amount,
which is how a lot of veterinary prescribing actually works.

**VET:** Right, let's get her on the scales first. Twenty-two point four
kilos. That's up from twenty-one last time, which is fine, she's still
growing.

**OWNER:** She eats constantly.

**VET:** Labradors do. Now, the itching — how long?

**OWNER:** Three weeks maybe? It's mostly her belly and her paws. She's
been chewing at the front paws especially.

**VET:** Let me look. Yes, there's redness between the pads on both
front feet, and the skin on the abdomen is inflamed, quite pink. No
obvious fleas but that doesn't rule them out. Any change of food?

**OWNER:** No, same as always.

**VET:** Okay. This looks like atopic dermatitis to me, an allergic skin
condition — the distribution is very typical, paws and belly. I want to
start her on ciclosporin. That's dosed by weight, five milligrams per
kilogram once daily, so for her that's a hundred and twelve milligrams,
and we round to the capsule size. Give it on an empty stomach, an hour
before food, because absorption drops if you give it with a meal.

**OWNER:** How long before it works?

**VET:** Four to six weeks for the full effect, so don't give up on it at
two. Come back in a month and we'll see where we are.

---

## 14 — Owner contradicts themselves

Tests handling of conflicting information within one consultation. The
corrected version is what belongs in the record.

**VET:** So when did the vomiting start?

**OWNER:** Yesterday morning.

**VET:** And how many times?

**OWNER:** Three, maybe four times. Actually — no, sorry, it was the day
before. Sunday. Because I remember I was doing the washing. So two days.

**VET:** Two days, and three or four episodes total, or per day?

**OWNER:** Total. Maybe five now I think about it.

**VET:** Is there food in it?

**OWNER:** The first one was food. After that it's been yellowy liquid.

**VET:** Right. Has he eaten anything he shouldn't have? Bins, socks,
anything from the garden?

**OWNER:** Not that I've seen. He does chew things though.

**VET:** Okay. Abdomen — let me feel. It's a bit tense here but he's not
crying out. Nothing that feels like an obstruction, no obvious foreign
body. Temperature's normal, thirty-eight point four. Gums are fine,
hydration seems reasonable.

**OWNER:** So what is it?

**VET:** Most likely a simple gastritis, but I'm not going to be
definitive when he's a dog who chews things. What I want is to withhold
food for twelve hours, then small bland meals — chicken and rice, tiny
portions, every few hours. If he vomits again after restarting food, or
if he becomes lethargic or the abdomen gets more painful, I need to see
him urgently for imaging.

---

## 15 — Rabbit, gut stasis

Tests a species other than dog or cat, with species-specific vocabulary.

**OWNER:** He hasn't eaten since yesterday evening and there's nothing in
the litter tray at all.

**VET:** Nothing at all? No droppings?

**OWNER:** A couple this morning but they were tiny. Much smaller than
normal.

**VET:** Right. With rabbits that's urgent, I'm glad you rang. Is he
moving around?

**OWNER:** He's just sitting hunched in the corner. He normally comes out
when I open the hutch.

**VET:** That hunched posture is a pain sign. Let me feel his abdomen —
yes, the stomach feels quite doughy and I can't feel normal gut movement.
Teeth — let me look at the back ones. There's some spurring on the lower
left molars, which could be why he stopped eating in the first place.

**OWNER:** Is it serious?

**VET:** Gut stasis in a rabbit is always serious, yes, because the gut
stopping causes more pain which stops them eating which stops the gut
further. We need to break that cycle today. I'm going to give him fluids
under the skin and start meloxicam for the pain, zero point six
milligrams per kilogram, once daily. He's two point one kilos so that's
one point three milligrams. And I want to syringe feed him — critical
care formula, every four hours, as much as he'll take.

**OWNER:** Through the night as well?

**VET:** Through the night as well, I'm afraid. And I want to see him
again tomorrow to burr those molar spurs down under sedation, assuming
he's stable enough by then.

---

## 16 — Interrupted, audio drops

Read this one with a deliberate pause where marked, or actually cut the
recording briefly. Tests whether the extraction stays honest about what
it could not hear rather than smoothing over the gap.

**VET:** — and how's his breathing been overnight?

**OWNER:** Worse I think. There's a sort of rasping when he breathes out.

**VET:** Let me listen. Okay, I can hear crackles in both lung fields,
worse on the left. Heart rate's elevated, one sixty. Gums are slightly
[PAUSE THREE SECONDS, OR CUT THE RECORDING HERE]

**OWNER:** — sorry, you cut out there.

**VET:** I said the gums are slightly tacky, so he's mildly dehydrated.
Look, I'm concerned about the lung sounds. I want chest x-rays today
before I commit to a treatment plan, because pneumonia and heart failure
look similar from the outside and the treatments are opposite.

**OWNER:** How long will that take?

**VET:** Leave him with us this morning. I'll ring you as soon as I've
seen the films.

---

## 17 — Two pets in one consultation

Tests whether the extraction stays with the presenting animal instead of
merging both. The record is for Poppy; Milo is context.

**OWNER:** I've brought both of them but it's really Poppy I'm worried
about.

**VET:** Okay, tell me about Poppy.

**OWNER:** She's had this discharge from one eye for about four days. The
left one. Milo's fine, he's just here because I couldn't leave him.

**VET:** Right, we'll focus on Poppy then. Any pawing at the eye?

**OWNER:** A bit of rubbing on the carpet.

**VET:** Let me have a look. The left eye's got a green-ish discharge and
the conjunctiva is quite red. Let me put some fluorescein in — that
stains any damage to the surface. No uptake, so no ulcer, that's the good
news. Right eye is clear.

**OWNER:** Should I be worried about Milo catching it?

**VET:** Keep an eye on him but it's not necessarily infectious. This
looks like a straightforward conjunctivitis. I'll give you chloramphenicol
ointment, small amount into the left eye four times a day for seven days.
Wash your hands between them.

**OWNER:** And if it doesn't clear?

**VET:** If it's no better in five days, back in. Don't let it run on.

---

## 18 — Emergency, distressed owner

Tests extraction from a chaotic, emotional transcript where the clinical
content is sparse and buried. Read this one fast, upset, overlapping.

**OWNER:** She's been hit by a car, she's in the back of the car now, I
don't know what to do —

**VET:** Okay, stay calm, I need you to answer some questions quickly. Is
she breathing?

**OWNER:** Yes, yes she's breathing but it's fast, it's really fast.

**VET:** Is she conscious? Is she responding to you?

**OWNER:** She lifted her head when I said her name. There's blood on her
back leg.

**VET:** Is the bleeding spurting or oozing?

**OWNER:** Oozing I think. Not spurting.

**VET:** Good. Do not move her more than you have to. If you have a towel
put gentle pressure on that leg. How far are you from us?

**OWNER:** Ten minutes.

**VET:** Come now. Drive carefully, I don't want two accidents. I'll have
the team ready and we'll take her straight through. Don't give her
anything, no food, no water, no painkillers of any kind.
