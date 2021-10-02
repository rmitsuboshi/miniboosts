

pub type Data<D> = Vec<D>;
pub type Label<L> = L;

pub type Sample<D, L> = Vec<(Data<D>, Label<L>)>;


pub fn to_sample<D, L>(examples: Vec<Data<D>>, labels: Vec<Label<L>>) -> Sample<D, L> {
    examples.into_iter()
            .zip(labels)
            .collect::<Sample<D, L>>()
}
